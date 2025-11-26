import sys
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MaskedSimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask):
        return torch.sum(x * (1 - mask), dim=1, keepdim=True)


def main() -> int:
    model = MaskedSimpleNet().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())

    batch_size = 10
    x = torch.randn(batch_size, 3).to(DEVICE)
    mask = torch.randint(0, 2, x.shape, dtype=torch.uint8).to(DEVICE)

    with torch.no_grad():
        y = model(x, mask)
        print("Input:\n", x)
        print("Mask:\n", mask)
        print("Output:\n", y)

    tm = torch.jit.trace(model.eval(), (x, mask))
    tm.save("MaskedNet.pt")

    print(model)
    print(f"Total params: {total_params}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
