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
    explicit Model(const std::string &model_path, bool auto_freeze = true)
        : model_(cms::torch::load(model_path)), device_(::torch::kCPU), auto_freeze_(auto_freeze) {
      model_.eval();
    }

    explicit Model(const std::string &model_path, ::torch::Device dev, bool auto_freeze = true)
        : model_(cms::torch::load(model_path, dev)), device_(dev), auto_freeze_(auto_freeze) {
      model_.eval();
    }

    // Move model to specified device memory space. Async load by specifying `non_blocking` (in default stream if not overridden by the caller)
    void to(::torch::Device dev, const bool non_blocking = false) {
      if (dev == device_)
        return;

      assert(!is_frozen_ && "Model is frozen, cannot be moved to another device!");
      model_.to(dev, non_blocking);
      device_ = dev;
      if (auto_freeze_) {
        freeze();
      }
    }

    void freeze() {
      if (!is_frozen_) {
        model_ = ::torch::jit::freeze(model_);
        is_frozen_ = true;
      }
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
    bool auto_freeze_;  // flag to indicate if the model should be automatically frozen after loading or moving to device
    bool is_frozen_ = false;  // flag to indicate if the model is frozen
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_Model_h
