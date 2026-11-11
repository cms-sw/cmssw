import sys
import torch
import torch.nn as nn

from onnx_utils import export


class ScaledNet(nn.Module):
    """Sum the three coordinates of each hit, and multiply the result by a scale shared by all the hits."""

    def __init__(self):
        super().__init__()

    def forward(self, positions, scale):
        # positions: [batch, 3], scale: [1], broadcast to all the hits
        return torch.sum(positions, dim=1, keepdim=True) * scale


def main() -> int:
    model = ScaledNet().eval()

    torch.manual_seed(42)
    batch_size = 10
    positions = torch.randn(batch_size, 3)
    scale = torch.tensor([1.5])

    with torch.no_grad():
        y = model(positions, scale)
        print("Positions:\n", positions)
        print("Scale:\n", scale)
        print("Output:\n", y)

    # [batch, 3] float, [1] float -> [batch, 1]; the scale has a fixed shape
    export(model, "ScaledNet.onnx", (positions, scale), ["positions", "scale"], ["result"], static=["scale"])

    print(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
