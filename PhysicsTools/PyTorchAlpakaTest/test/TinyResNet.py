import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(16)

        self.layer1 = ResidualBlock(16)
        self.layer2 = ResidualBlock(16)

        self.conv_down = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.layer3 = ResidualBlock(32)
        self.layer4 = ResidualBlock(32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv_down(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def main() -> int:
    model = MiniResNet(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())

    batch_size = 1
    x = torch.randn(batch_size, 3, 9, 9).to(DEVICE)

    with torch.no_grad():
        y = model(x)
        print("Input:\n", x)
        print("Output:\n", y)

    tm = torch.jit.trace(model.eval(), x)
    tm.save("TinyResNet.pt")

    print(model)
    print(f"Total params: {total_params}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
