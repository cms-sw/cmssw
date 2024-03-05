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
    def __init__(self, num_classes=10):
        super(Simple1DNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 128, 128) # Assuming input length is 256, after pooling it becomes 128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 128) # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
model = MyModuleSum()
model(torch.ones(1<<16), torch.ones(1<<16))

tm = torch.jit.trace(model.eval(), [torch.ones(1<<16)])

tm.save(f"{datadir}/simple_dnn_largeinput.pt")

print("simple_dnn_largeinput.pt created successfully!")
