---
title: Deep Q Network(DQN)
date: 2021-12-07
categories: RL
tag: Machine Learning
---
**如今, 随着机器学习在日常生活中的各种应用, 各种机器学习方法也在融汇, 合并, 升级. 而我们今天所要探讨的强化学习则是这么一种融合了神经网络和 Q learning 的方法, 名字叫做 Deep Q Network. 这种新型结构是为什么被提出来呢? 原来, 传统的表格形式的强化学习有这样一个瓶颈。**
<!--more-->
## QLearning
QLearning是强化学习算法中value-based的算法，Q即为Q（s,a）就是在某一时刻的 s 状态下(s∈S)，采取 动作a (a∈A)动作能够获得收益的期望，环境会根据agent的动作反馈相应的回报reward r，所以算法的主要思想就是将State与Action构建成一张Q-table来存储Q值，然后根据Q值来选取能够获得最大的收益的动作。

| Q-Table | a1 | a2 |
| ----: | ----: | ----: |
| s1 | q(s1,a1) | q(s1,a2) |
| s2 | q(s2,a1) | q(s2,a2) |
| s3 | q(s3,a1) | q(s3,a2) |

Qlearning的主要优势就是使用了时间差分法TD（融合了蒙特卡洛和动态规划）能够进行离线学习, 使用bellman方程可以对马尔科夫过程求解最优策略。
Q(s,a)状态动作值函数

![20211207150858](https://cdn.jsdelivr.net/gh/wxt406611016/cdn/image/20211207150858.png)

其中Gt是从t时刻开始的总折扣奖励，从这里我们能看出来 γ衰变值对Q函数的影响，γ越接近于1代表它越有远见会着重考虑后续状态的的价值，当γ接近0的时候就会变得近视只考虑当前的利益的影响。所以从0到1，算法就会越来越会考虑后续回报的影响。

---
## 为什么使用神经网络(为什么提出DQN)
我们使用表格来存储每一个状态 state, 和在这个 state 每个行为 action 所拥有的 Q 值. 而当今问题是在太复杂, 状态可以多到比天上的星星还多(比如下围棋). 如果全用表格来存储它们, 恐怕我们的计算机有再大的内存都不够, 而且每次在这么大的表格中搜索对应的状态也是一件很耗时的事. 不过, 在机器学习中, 有一种方法对这种事情很在行, 那就是神经网络. 我们可以将状态和动作当成神经网络的输入, 然后经过神经网络分析后得到动作的 Q 值, 这样我们就没必要在表格中记录 Q 值, 而是直接使用神经网络生成 Q 值. 还有一种形式的是这样, 我们也能只输入状态值, 输出所有的动作值, 然后按照 Q learning 的原则, 直接选择拥有最大值的动作当做下一步要做的动作. 我们可以想象, 神经网络接受外部的信息, 相当于眼睛鼻子耳朵收集信息, 然后通过大脑加工输出每种动作的值, 最后通过强化学习的方式选择动作.

---
## 如何更新神经网络
接下来我们基于第二种神经网络来分析, 我们知道, 神经网络是要被训练才能预测出准确的值. 那在强化学习中, 神经网络是如何被训练的呢? 首先, 我们需要 a1, a2 正确的Q值, 这个 Q 值我们就用之前在 Q learning 中的 Q 现实来代替. 同样我们还需要一个 Q 估计 来实现神经网络的更新. 所以神经网络的的参数就是老的 NN 参数 加学习率 alpha 乘以 Q 现实 和 Q 估计 的差距. 我们整理一下.

![20211207151445](https://cdn.jsdelivr.net/gh/wxt406611016/cdn/image/20211207151445.png)

我们通过 NN 预测出Q(s2, a1) 和 Q(s2,a2) 的值, 这就是 Q 估计. 然后我们选取 Q 估计中最大值的动作来换取环境中的奖励 reward. 而 Q 现实中也包含从神经网络分析出来的两个 Q 估计值, 不过这个 Q 估计是针对于下一步在 s' 的估计. 最后再通过刚刚所说的算法更新神经网络中的参数.

---
## DQN在CartPole-v0上的应用
首先看模型未训练前效果：

<video src="https://cdn.jsdelivr.net/gh/wxt406611016/cdn@master/vedio/20211208-202305.mp4" width="400px" height="300px" controls="controls"></video>

对应的奖励如下所示，可以看到毫无policy可言，效果一塌糊涂：
```
reward_sum:3.27
reward_sum:3.96
reward_sum:1.72
reward_sum:2.46
reward_sum:2.01
reward_sum:2.98
reward_sum:3.74
reward_sum:3.25
reward_sum:2.85
```
---
为了改善这个情况，接下来需要训练模型。

训练模型：

环境
* python3
* torch pytorch #用于创建神经网络
* numpy 
* gym #游戏包

main.py :（训练网络）
```
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym                                      # 导入gym

# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]       # 杆子状态个数 (4个)

#定义Net类（定义网络）
class Net(nn.Module):
    def __init__(self):                                                         
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                            

        self.fc1 = nn.Linear(N_STATES, 50)                                      
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 50)                                      
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)                                     
        self.out.weight.data.normal_(0, 0.1)                                   
    def forward(self, x):                                                       
        x = F.relu(self.fc1(x))                                                 
        x = F.relu(self.fc2(x))
        return self.out(x)                                                    

# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                                           # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])                         # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))    # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])             # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])                       # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)                              # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()                                 # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)           # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)                                 # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                              # 清空上一步的残余更新参数值
        loss.backward()                                                         # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                                   # 更新评估网络的所有参数

dqn = DQN()                                                             # 令dqn=DQN类
score = deque(maxlen=50)
for i in range(4000):                                                    # 400个episode循环
    s = env.reset()                                                     # 重置环境
    episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
    while True:                                                         # 开始一个episode (每一个循环代表一步)
        env.render()                                                    # 显示实验动画
        a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2
        dqn.store_transition(s, a, new_r, s_)                 # 存储样本
        episode_reward_sum += new_r                           # 逐步加上一个episode内每个step的reward
        s = s_                                                # 更新状态
        if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000，开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()
        if done:       # 如果done为True
            score.append(episode_reward_sum)
            print(f'\r==> episode: {i}---reward_mean100: {round(np.mean(score), 2)}',end="")
            if i % 100 == 0:
                print(f'\r==> episode: {i}---reward_mean100: {round(np.mean(score), 2)}')
            break
    if np.mean(score) > 600 and i > 500:
        torch.save(dqn.eval_net,'dqn_eval.pth')
        break
print('------FINISHED------')
```
训练过程中终端输出如下：
```
==> episode: 0---reward_mean100: 2.87
==> episode: 100---reward_mean100: 2.47
==> episode: 200---reward_mean100: 2.58
==> episode: 300---reward_mean100: 350.95
==> episode: 400---reward_mean100: 403.62
==> episode: 500---reward_mean100: 524.56
==> episode: 518---reward_mean100: 555.81
------FINISHED------
```
可以看出在reward_mean大于550时程序停止，同时得到dqn_eval.pth文件，可用于测试训练的效果，测试代码如下所示：

test.py : （测试模型学习的效果）
```
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym                                      # 导入gym

# 超参数
EPSILON = 0.9                                   # greedy policy
env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]       # 杆子状态个数 (4个)

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                            

        self.fc1 = nn.Linear(N_STATES, 50)                                      
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 50)                                      
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)                                     
        self.out.weight.data.normal_(0, 0.1)                                   
    def forward(self, x):                                                       
        x = F.relu(self.fc1(x))                                                 
        x = F.relu(self.fc2(x))
        return self.out(x)

# 定义DQN类 (定义两个网络)
class DQN(object):
  def __init__(self):                                                         # 定义DQN的一系列属性
    self.eval_net = torch.load('./CartPole-v0/dqn_eval_v3.pth')
    self.eval_net.eval()

  def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
    x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
    if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
      actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
      action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
      action = action[0]                                                  # 输出action的第一个数
    else:                                                                   # 随机选择动作
      action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
    return action                                                           # 返回选择的动作 (0或1)


dqn = DQN()                                                             # 令dqn=DQN类
while True:
  s = env.reset()                                                     # 重置环境
  episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
  while True:                                                         # 开始一个episode (每一个循环代表一步)
    env.render()                                                    # 显示实验动画
    a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
    s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
    x, x_dot, theta, theta_dot = s_
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    new_r = r1 + r2
    episode_reward_sum += new_r                           
    s = s_                                                
    if done:
      print(f'reward_sum:{round(episode_reward_sum, 2)}')
      break
```
动画效果如下，测试的获得奖励情况在终端中有显示，如下:

<video src="https://cdn.jsdelivr.net/gh/wxt406611016/cdn@master/vedio/20211208-204030.mp4" width="400px" height="300px" controls="controls"></video>

```
reward_sum:237.14
```

jupyter代码在github上可以获取，[点击传送至Github](https://github.com/wxt406611016/DQN-CartPole-v0)