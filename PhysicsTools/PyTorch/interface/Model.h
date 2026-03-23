#ifndef PhysicsTools_PyTorch_interface_Model_h
#define PhysicsTools_PyTorch_interface_Model_h

#include <string>
#include <vector>

#include "PhysicsTools/PyTorch/interface/ScriptModuleLoad.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

namespace cms::torch {

  // Wrapper of torch::jit::script::Module:
  // - https://docs.pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#class-module
  class Model {
  public:
    explicit Model(const std::string &model_path) : model_(cms::torch::load(model_path)), device_(::torch::kCPU) {}

    explicit Model(const std::string &model_path, ::torch::Device dev)
        : model_(cms::torch::load(model_path, dev)), device_(dev) {}

    // Move model to specified device memory space. Async load by specifying `non_blocking` (in default stream if not overridden by the caller)
    void to(::torch::Device dev, const bool non_blocking = false) {
      if (dev == device_)
        return;
      model_.to(dev, non_blocking);
      device_ = dev;
    }

    // Forward pass (inference) of model, returns torch::IValue (multi output support). Match native torchlib interface.
    ::torch::IValue forward(std::vector<::torch::IValue> &inputs) {
      // Disabling autograd
      ::torch::NoGradGuard no_grad_guard;
      return model_.forward(inputs);
    }

    // Get model current device information.
    ::torch::Device device() const { return device_; }

  protected:
    ::torch::jit::script::Module model_;  // underlying JIT model
    ::torch::Device device_;              // device where the model is allocated (default CPU)
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_Model_h
