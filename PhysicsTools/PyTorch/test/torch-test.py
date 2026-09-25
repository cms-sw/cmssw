#!/usr/bin/env python3
import sys
import torch
from torch_utils import check_torch_gpu
gpu, gpu_device, gpu_name = check_torch_gpu(torch, sys.argv[1])
if not gpu:
  exit(1)

x = torch.rand(10000, 10000, device=gpu_device)
y = torch.rand(10000, 10000, device=gpu_device)
z = torch.mm(x, y)
print("OK. Computed on", gpu_name, ", Result:", z[0][0].item())
