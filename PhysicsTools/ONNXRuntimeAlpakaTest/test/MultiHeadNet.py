import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnx_utils import export


class MultiHeadNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[32, 64, 256, 64, 32],
                 reg_output_dim=1, class_output_dim=3):
        super().__init__()
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

        self.reg_head = nn.Linear(in_dim, reg_output_dim)
        nn.init.constant_(self.reg_head.weight, 0.03)
        nn.init.constant_(self.reg_head.bias, 0.1)

        self.class_head = nn.Linear(in_dim, class_output_dim)
        with torch.no_grad():
            self.class_head.weight[:] = torch.tensor([[0.06, 0.07, 0.08]] * in_dim).T
            self.class_head.bias[:] = torch.tensor([0.1, 0.2, 0.3])

    def forward(self, x):
        features = self.backbone(x)
        reg_out = self.reg_head(features)
        class_out = self.class_head(features)
        class_out = F.softmax(class_out, dim=1)
        return reg_out, class_out


def main() -> int:
    params = {
        "input_dim": 3,
        "hidden_dims": [32, 64, 256, 64, 32],
        "reg_output_dim": 1,
        "class_output_dim": 3
    }

    multiheadnet = MultiHeadNet(**params).eval()
    total_params = sum(p.numel() for p in multiheadnet.parameters())
    print(f"Total params: {total_params}")

    batch_size = 32
    torch.manual_seed(42)  # deterministic input
    input_tensor = torch.rand(batch_size, params["input_dim"])

    with torch.no_grad():
        reg_out, class_out = multiheadnet(input_tensor)
        print("Input: ", input_tensor)
        print("Regression:\n", reg_out)
        print("Classification (softmax):\n", class_out)

    # [batch, 3] -> [batch, 1], [batch, 3]
    export(multiheadnet, "MultiHeadNet.onnx", (input_tensor,), ["particles"],
           ["regression_head", "classification_head"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
