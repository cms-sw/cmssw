# @see: https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser

if not torch.__version__.startswith("2.6."):
    print(
        "This script recommends PyTorch 2.6.x, due to breaking compatibility changes in AOT Inductor API, ",
        "but found version {torch.__version__}."
        "Please install the correct version of PyTorch in case of compilation errors with CMSSW environment."
        "To see the version supported by your CMSSW release, run `scram b tool info pytorch`."
    )

class RegressionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(RegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        # repro: weights zero and bias to 0.5
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0.5)
    
    def forward(self, input):
        x = self.fc(input)
        return x
    

class ClassifierModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(ClassifierModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Fully connected layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation
        self.fc.weight.data = torch.tensor(
            [
                [1.0, 1.0, 0.0], 
                [1.0, 1.0, 0.0]
            ]
        )
        self.fc.bias.data = torch.tensor([0.0, 0.0])

        # note: Prevent updates
        self.fc.weight.requires_grad = False
        self.fc.bias.requires_grad = False
    
    def forward(self, input):
        x = self.fc(input)  # Linear transformation
        x = self.softmax(x)  # Apply softmax
        return x
    

def parse_args():
    parser = ArgumentParser(description="AOT Inductor Export Example")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["regression", "classification"], 
        required=True,
        help="Choose the model type to export: 'regression' or 'classification'")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Target device for export.")
    parser.add_argument(
        "--output", 
        type=str, 
        default="model", 
        help="Name for the output file.")
    return parser.parse_args()


def get_model(model_type, device):
    if model_type == "regression":
        return RegressionModel().to(device).eval()
    elif model_type == "classification":
        return ClassifierModel().to(device).eval()
    raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")


def get_input_tensor(model_type, device):
    if model_type == "regression":
        return torch.randn(32, 3, device=device)
    elif model_type == "classification":
        return torch.randn(32, 3, device=device)
    raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")


def aoti_compile_and_package(model_type, device, package_name=None, inductor_configs=None):
    model = get_model(model_type, device)
    with torch.no_grad():
        example_inputs=(get_input_tensor(model_type, device),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        # [Optional] Specify the first dimension of the input x as dynamic.
        exported = torch.export.export(model, example_inputs, dynamic_shapes={"input": {0: batch_dim}})
        # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
        # Depending on your use case, e.g. if your training platform and inference platform
        # are different, you may choose to save the exported model using torch.export.save and
        # then load it back using torch.export.load on your inference platform to run AOT compilation.
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            # [Optional] Specify the generated shared library path. If not specified,
            # the generated artifact is stored in your system temp directory.
            package_path=os.path.join(os.getcwd(), f"{package_name}.pt2"),
            inductor_configs=inductor_configs
        )


def main():
    args = parse_args()
    aoti_compile_and_package(
        model_type=args.model,
        device=args.device, 
        package_name=args.output,
        inductor_configs={
            "aot_inductor.package_cpp_only": True
        }
    )
    

if __name__ == "__main__":
    sys.exit(main())
