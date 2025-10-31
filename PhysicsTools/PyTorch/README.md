# PhysicsTools/PyTorch
The torch interface is split into a general torch wrapper and an [Alpaka supported interface](../PyTorchAlpaka). A full ML CMSSW pipeline is implemented and tested in [PyTorchAlpakaTest](../PyTorchAlpakaTest) and serves as a tutorial how to run direct inference with `Portable` modules.

This package enables seamless integration between PyTorch and the CMSSW SoA implementation. It provides:
- Support for automatic conversion of optimized SoA to torch tensors, with memory blobs reusage.
- Support for both just-in-time (JIT) and ahead-of-time (AOT) model execution (Beta version for AOT).

## PyTorchService
To not interfere with CMSSW threading model, `PyTorchService` **MUST** be included in the `cmsRun` configuration path, whenever PyTorch is used. The service will disable internal threading of PyTorch 
An example setup can be found in [PyTorchAlpakaTest](../PyTorchAlpakaTest/test/testPyTorchAlpakaHeterogeneousPipeline.py). More on PyTorch threading model: [CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).

## Inference: JIT and AOT Model Execution
A Wrapper for the torch model stored with Just-in-Time (JIT), `Model` class, is provided enabling inference with native torch `Tensor` objects. To run direct inference on CMSSW PortableObjects and SoAs, the custom wrapper (see: [AlpakaModel.h](../PyTorchAlpaka/interface/alpaka/AlpakaModel.h)) has to be used.

### Just-in-Time:
- Loads `torch::jit::script::Module` at runtime.
- Compiles model on-the-fly.
- Introduces warm-up overhead without additional optimization.
- When storing model through tracing, compatibility and correctness have to be checked

Example how to export models from PyTorch Python API, more can be found in [PyTorchAlpakaTest/python](../PyTorchAlpakaTest/python/):
```py
batch_size = 10
input_tensor = torch.randn(batch_size, shape)
tm = torch.jit.trace(simplenet.eval(), input_tensor)
tm.save(f"traced_model.pt")
```

### Ahead-of-Time (beta version not production ready):
- Uses PyTorch AOT compiler to generate `.cpp` and `.so` files. (prerequisite done manually by end-user)
- Package provide helper scripts to automate compilation process with CMSSW provided tools to some extent
- Loads compiled model via [AOTIModelPackageLoader](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_package/model_package_loader.h).
- Eliminates JIT overhead, enable optimization, but requires architecture-specific handling 

More in depth introduction to the concepts used with AOT compilation see: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
