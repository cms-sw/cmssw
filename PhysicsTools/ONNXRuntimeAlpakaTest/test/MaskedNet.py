import sys
import torch
import torch.nn as nn

from onnx_utils import export


class MaskedSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return torch.sum(x * (1 - mask), dim=1, keepdim=True)


def main() -> int:
    model = MaskedSimpleNet().eval()

    torch.manual_seed(42)
    batch_size = 10
    x = torch.randn(batch_size, 3)
    mask = torch.randint(0, 2, x.shape, dtype=torch.uint8)

    with torch.no_grad():
        y = model(x, mask)
        print("Input:\n", x)
        print("Mask:\n", mask)
        print("Output:\n", y)

    # [batch, 3] float, [batch, 3] uint8 -> [batch, 1]
    export(model, "MaskedNet.onnx", (x, mask), ["particles", "mask"], ["regression_head"])

    print(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
