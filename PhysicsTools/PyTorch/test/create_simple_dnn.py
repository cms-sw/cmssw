import torch

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

print(tm.graph)

tm.save("simple_dnn.pt")
