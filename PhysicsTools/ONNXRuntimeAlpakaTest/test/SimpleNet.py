import sys
import torch
import torch.nn as nn

from onnx_utils import export, FeatureMajor


class SimpleNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 128], output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> int:
    params = {
        "input_dim": 3,
        "output_dim": 1,
        "hidden_dims": [32, 64, 256, 64, 32],
    }

    torch.manual_seed(42)
    simplenet = SimpleNet(**params).eval()
    print(simplenet)

    total_params = sum(p.numel() for p in simplenet.parameters())
    print(f"Total params: {total_params}")

    batch_size = 10
    input_tensor = torch.rand(batch_size, params["input_dim"])
    with torch.no_grad():
        output_tensor = simplenet(input_tensor)
        print("Output shape: ", output_tensor.shape)

    # [batch, 3] -> [batch, 1]
    export(simplenet, "SimpleNet.onnx", (input_tensor,), ["particles"], ["regression_head"])
    # [3, batch] -> [1, batch], to use the SoA memory directly with Layout::FeatureMajor
    export(FeatureMajor(simplenet), "SimpleNetFeatureMajor.onnx", (input_tensor.T.contiguous(),), ["particles"],
           ["regression_head"], batch_axis=1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
