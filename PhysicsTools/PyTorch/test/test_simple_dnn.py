#!/usr/bin/env python3

import sys
import os
import torch
from torch_utils import check_torch_gpu
gpu, device, gpu_name = check_torch_gpu(torch, sys.argv[1])
if not gpu:
  exit(1)

if len(sys.argv)>=3:
    datadir=sys.argv[2]
else:
    thisdir=os.path.dirname(os.path.abspath(__file__))
    datadir=os.path.join(os.path.dirname(thisdir), "bin", "data")

ptfile = os.path.join(datadir, "simple_dnn.pt")
print("loading:", ptfile)

tm = torch.jit.load(ptfile)
tm.eval()

# dummy input (same shape used during trace: 10)
x = torch.ones(10)

# optional: run on gpu if available
tm.to(device)
x = x.to(device)

with torch.no_grad():
    y = tm(x)

print("ok. output:", y.item())
print("device:", gpu_name)
