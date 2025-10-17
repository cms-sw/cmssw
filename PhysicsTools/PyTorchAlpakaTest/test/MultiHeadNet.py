import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[32, 64, 256, 64, 32], 
                 reg_output_dim=1, class_output_dim=3):
        super().__init__()
        # Shared backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            nn.init.constant_(linear.weight, 0.025)
            nn.init.constant_(linear.bias, 0.1)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Regression head
        self.reg_head = nn.Linear(in_dim, reg_output_dim)
        nn.init.constant_(self.reg_head.weight, 0.03)
        nn.init.constant_(self.reg_head.bias, 0.1)

        # Classification head: different values for each class
        self.class_head = nn.Linear(in_dim, class_output_dim)
        # e.g., small differences per output neuron
        with torch.no_grad():
            self.class_head.weight[:] = torch.tensor([[0.06, 0.07, 0.08]] * in_dim).T
            self.class_head.bias[:] = torch.tensor([0.1, 0.2, 0.3])

    def forward(self, x):
        features = self.backbone(x)
        reg_out = self.reg_head(features)
        class_out = self.class_head(features)
        # Apply softmax along class dimension
        class_out = F.softmax(class_out, dim=1)
        return reg_out, class_out


def main() -> int:
    params = {
        "input_dim": 3,
        "hidden_dims": [32, 64, 256, 64, 32],
        "reg_output_dim": 1,
        "class_output_dim": 3
    }
    
    multiheadnet = MultiHeadNet(**params).to(DEVICE)
    total_params = sum(p.numel() for p in multiheadnet.parameters())
    print(f"Total params: {total_params}")

    batch_size = 32
    torch.manual_seed(42)  # deterministic input
    input_tensor = torch.rand(batch_size, params["input_dim"]).to(DEVICE)

    with torch.no_grad():
        reg_out, class_out = multiheadnet(input_tensor)
        print("Input: ", input_tensor)
        print("Regression:\n", reg_out)
        print("Classification (softmax):\n", class_out)

    tm = torch.jit.trace(multiheadnet.eval(), input_tensor)
    tm.save(f"MultiHeadNet.pt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
