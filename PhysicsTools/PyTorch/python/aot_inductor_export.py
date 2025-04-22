# @see: https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html

import os
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        # repro: weights zero and bias to 0.5
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0.5)
    
    def forward(self, input):
        x = self.fc(input)
        return x
    

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs=(torch.randn(8, 3, device=device),)
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
        package_path=os.path.join(os.getcwd(), f"model_{device}.pt2"),
        inductor_configs={"aot_inductor.package_cpp_only": True}
    )