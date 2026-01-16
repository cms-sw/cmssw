#!/usr/bin/env python3

import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("device name:", torch.cuda.get_device_name() if torch.cuda.is_available() else None)

# small compute test
if torch.cuda.is_available():
    x = torch.rand(10000, 10000, device="cuda")
    y = torch.rand(10000, 10000, device="cuda")
    z = torch.mm(x, y)
    print("OK. Computed on CUDA. Result:", z[0][0].item())
else:
    print("NO CUDA")
    exit(1)
