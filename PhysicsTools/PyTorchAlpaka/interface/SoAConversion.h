#ifndef PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h
#define PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h

#include <cstdint>
#include <vector>

#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorHandle.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorRegistry.h"

namespace cms::torch::alpakatools {

  std::vector<::torch::IValue> convertInput(const TensorRegistry& input, ::torch::Device device);

  ::torch::Tensor convertOutput(const TensorRegistry& output, ::torch::Device device);
  void convertOutput(const ::torch::IValue& tensors, const TensorRegistry& output, ::torch::Device device);
  void convertOutput(const std::vector<::torch::IValue>& tensors, const TensorRegistry& output, ::torch::Device device);

  ::torch::Tensor arrayToTensor(::torch::Device device, const PortableTensorHandle& view);

  // AOT specific implementations
  // std::vector<::torch::Tensor> convertInputTensor(const ModelMetadata& metadata, ::torch::Device device);
  // void convertOutput(const std::vector<::torch::Tensor>& tensors, const TensorRegistry& output, ::torch::Device device);

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h
