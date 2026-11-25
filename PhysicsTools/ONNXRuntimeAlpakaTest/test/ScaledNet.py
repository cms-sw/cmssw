import sys
import torch
import torch.nn as nn

from onnx_utils import export


class ScaledNet(nn.Module):
    """Sum the three coordinates of each hit, and multiply the result by a scale shared by all the hits.
    Return also the square of the scale, as a [1] tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, positions, scale):
        # positions: [batch, 3], scale: [1], broadcast to all the hits
        return torch.sum(positions, dim=1, keepdim=True) * scale, scale * scale


def main() -> int:
    model = ScaledNet().eval()

    torch.manual_seed(42)
    batch_size = 10
    positions = torch.randn(batch_size, 3)
    scale = torch.tensor([1.5])

    with torch.no_grad():
        y, squared_scale = model(positions, scale)
        print("Positions:\n", positions)
        print("Scale:\n", scale)
        print("Output:\n", y)
        print("Squared scale:\n", squared_scale)

    # [batch, 3] float, [1] float -> [batch, 1] float, [1] float; the scale and its square have a fixed shape
    export(model, "ScaledNet.onnx", (positions, scale), ["positions", "scale"], ["result", "squared_scale"],
           static=["scale", "squared_scale"])

    print(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
