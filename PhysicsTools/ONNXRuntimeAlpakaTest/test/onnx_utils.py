import torch


def export(model, path, inputs, input_names, output_names, batch_axis=0, static=()):
    """Export a model to ONNX, with a dynamic batch size along `batch_axis` for all inputs and outputs,
    except those listed in `static`, e.g. the [1] tensors used for SOA_SCALARs."""
    dynamic_axes = {name: {batch_axis: "batch"} for name in input_names + output_names if name not in static}
    torch.onnx.export(model.eval(), inputs, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    print(f"Exported {path}: inputs {input_names}, outputs {output_names}")


class FeatureMajor(torch.nn.Module):
    """Wrap a model to accept and return feature-major (transposed) tensors, for Layout::FeatureMajor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.transpose(0, 1)).transpose(0, 1)
