#include "PhysicsTools/PyTorch/interface/ScriptModuleLoad.h"

namespace cms::torch {

  // `torch::jit::load` wrapper to load a JIT exported TorchScript model.
  // In case of failure, a cms::Exception is thrown with context and error details.
  // `dev` optional device to load the model on. Async load is not supported. Use model.to(device, true) instead.
  // TODO: figure out how to fix linker error without inlining and adding pytorch-cuda dependency (move impl to src/)
  //       that brings CUDA headers in alpaka ROCm/HIP compiled products.
  ScriptedModule load(const std::string &model_path, std::optional<::torch::Device> dev) {
    ScriptedModule model;
    try {
      model = ::torch::jit::load(model_path, dev);
    } catch (const c10::Error &e) {
      cms::Exception ex("ModelLoadingError");
      ex << "Error loading the model from path: " << model_path << "\n"
         << "Details: " << e.what();
      ex.addContext("Calling cms::torch::load(const std::string&)");
      throw ex;
    }
    return model;
  }

}  // namespace cms::torch
