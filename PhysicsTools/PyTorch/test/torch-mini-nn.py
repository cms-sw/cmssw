#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
if (device == "cpu") and (len(sys.argv) > 1) and (sys.argv[1] != "cpu"):
  pritn("Unable to find accelerator",sys.argv[1])
  exit(1)

# simple fully connected network
model = nn.Sequential(
    nn.Linear(1000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 1)
).to(device)

# random data
x = torch.randn(5000, 1000, device=device)
y = torch.randn(5000, 1, device=device)

opt = optim.Adam(model.parameters(), lr=1e-3)

# train 5 steps
for i in range(5):
    opt.zero_grad()
    pred = model(x)
    loss = ((pred - y)**2).mean()
    loss.backward()
    opt.step()
    print("step:", i, "loss:", loss.item())

