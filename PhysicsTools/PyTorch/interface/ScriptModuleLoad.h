#ifndef PhysicsTools_PyTorch_interface_ScriptModuleLoad_h
#define PhysicsTools_PyTorch_interface_ScriptModuleLoad_h

#include <optional>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace cms::torch {

  using ScriptedModule = ::torch::jit::script::Module;

  // `torch::jit::load` wrapper to load a JIT exported TorchScript model.
  // In case of failure, a cms::Exception is thrown with context and error details.
  // `dev` optional device to load the model on. Async load is not supported. Use model.to(device, true) instead.
  // TODO: figure out how to fix linker error without inlining and adding pytorch-cuda dependency (move impl to src/)
  //       that brings CUDA headers in alpaka ROCm/HIP compiled products.
  inline ScriptedModule load(const std::string &model_path, std::optional<::torch::Device> dev = std::nullopt) {
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

#endif  // PhysicsTools_PyTorch_interface_ScriptModuleLoad_h