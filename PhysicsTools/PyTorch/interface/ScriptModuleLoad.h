#ifndef PhysicsTools_PyTorch_interface_ScriptModuleLoad_h
#define PhysicsTools_PyTorch_interface_ScriptModuleLoad_h

#include <optional>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

namespace cms::torch {

  using ScriptedModule = ::torch::jit::script::Module;

  // `torch::jit::load` wrapper to load a JIT exported TorchScript model.
  // In case of failure, a cms::Exception is thrown with context and error details.
  // `dev` optional device to load the model on. Async load is not supported. Use model.to(device, true) instead.
  ScriptedModule load(const std::string &model_path, std::optional<::torch::Device> dev = std::nullopt);

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ScriptModuleLoad_h
