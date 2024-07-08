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

class MyModuleSum(torch.nn.Module):
    def __init__(self):
        super(MyModuleSum, self).__init__()
        
    def forward(self, A, B):
          return A+B

      
model = MyModuleSum()
model(torch.ones(1<<16), torch.ones(1<<16))

tm = torch.jit.trace(model.eval(), [torch.ones(1<<16), torch.ones(1<<16)])

tm.save(f"{datadir}/simple_dnn_sum.pt")

print("simple_dnn_largeinput.pt created successfully!")
