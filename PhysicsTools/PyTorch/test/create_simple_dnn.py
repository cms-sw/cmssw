#!/usr/bin/env python3

import sys
import os
import torch

# prepare the datadir
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

os.makedirs(datadir, exist_ok=True)

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(N, M))
        self.bias = torch.nn.Parameter(torch.ones(N))
        
    def forward(self, input):
          return torch.sum(torch.nn.functional.elu(self.weight.mv(input) + self.bias))


module = MyModule(10, 10)
x = torch.ones(10)

tm = torch.jit.trace(module.eval(), x)

tm.save(f"{datadir}/simple_dnn.pt")

print("simple_dnn.pt created successfully!")
