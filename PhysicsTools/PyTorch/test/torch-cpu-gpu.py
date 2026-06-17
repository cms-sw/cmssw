#!/usr/bin/env python3
import sys
import time
import torch
from torch_utils import check_torch_gpu

def test(device, name="CPU"):
    print("Testing on %s (%s)" % (name, device))
    x = torch.rand(8000, 8000, device=device)
    y = torch.rand(8000, 8000, device=device)

    torch.cuda.synchronize() if device=="cuda" else None
    t0 = time.time()
    torch.mm(x, y)
    torch.cuda.synchronize() if device=="cuda" else None
    print("Time:", time.time()-t0, "seconds")

gpu, gpu_device, gpu_name = check_torch_gpu(torch, sys.argv[1])
if not gpu:
  exit(1)

test("cpu")
if gpu_device != "cpu":
    test(gpu_device, gpu_name)
