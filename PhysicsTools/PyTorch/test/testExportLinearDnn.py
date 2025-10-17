import sys
import os
import torch

if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

os.makedirs(datadir, exist_ok=True)

class LinearModel(torch.nn.Module):
    def __init__(self, N, M):
        super(LinearModel, self).__init__()
        
        self.linear = torch.nn.Linear(N, M, bias=False)
        with torch.no_grad():
            self.linear.weight.data[...] = torch.Tensor([[-0.1, 0.2, 2], [0.1, -2.3, 4.0]])
        
    def forward(self, x):
        x = self.linear(x)
        return x


module = LinearModel(3, 2)
x = torch.tensor([[1., 2., 1.], [2., 4., 3.], [3., 4., 1.], [2., 3., 2.]])

tm = torch.jit.trace(module.eval(), x)
tm.save(f"{datadir}/linear_dnn.pt")
