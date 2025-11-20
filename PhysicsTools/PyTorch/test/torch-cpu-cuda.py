#!/usr/bin/env python3

import time
import torch

def test(device):
    print("Testing on", device)
    x = torch.rand(8000, 8000, device=device)
    y = torch.rand(8000, 8000, device=device)

    torch.cuda.synchronize() if device=="cuda" else None
    t0 = time.time()
    torch.mm(x, y)
    torch.cuda.synchronize() if device=="cuda" else None
    print("Time:", time.time()-t0, "seconds")

if torch.cuda.is_available():
    test("cpu")
    test("cuda")
else:
    print("no cuda")
    exit(1)
