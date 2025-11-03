#ifndef PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h
#define PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h

#include <cstdint>
#include <vector>

#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorRegistry.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorHandle.h"

namespace cms::torch::alpakatools {

  template <typename TQueue>
  inline ::torch::Tensor arrayToTensor(::torch::Device device, ITensorHandle<TQueue>& tensor_handle) {
    // const_cast is required as `from_blob` does not take const pointer even if it does not modify the data
    // see: https://discuss.pytorch.org/t/using-torch-from-blob-with-const-data/141597
    // https://github.com/pytorch/pytorch/blob/89a6dbe73af4ca64ee26f4e46219e163b827e698/aten/src/ATen/ops/from_blob.h#L107
    //
    // TODO: open issue to `pytorch` repo:
    //  - see if they can add const correctness, or get to know why const is currently prevented?
    assert(reinterpret_cast<intptr_t>(tensor_handle.data()) % tensor_handle.alignment() == 0);
    auto options = ::torch::TensorOptions().dtype(tensor_handle.type()).device(device).pinned_memory(true);
    return ::torch::from_blob(tensor_handle.data(), tensor_handle.sizes(), tensor_handle.strides(), options);
  }

  template <typename TQueue>
  inline std::vector<::torch::IValue> convertInput(TensorRegistry<TQueue>& inputs, ::torch::Device device) {
    std::vector<::torch::IValue> tensors(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      tensors[i] = arrayToTensor(device, inputs[i]);
    }
    return tensors;
  }

  template <typename TQueue>
  inline ::torch::Tensor convertOutput(TensorRegistry<TQueue>& outputs, ::torch::Device device) {
    return arrayToTensor(device, outputs[0]);
  }

  template <typename TQueue>
  inline void convertOutput(const ::torch::IValue& tensor, TensorRegistry<TQueue>& outputs, ::torch::Device device) {
    if (tensor.isTuple()) {
      const auto tensor_tuple = tensor.toTuple();
      for (size_t i = 0; i < outputs.size(); i++) {
        arrayToTensor(device, outputs[i]) = tensor_tuple->elements()[i].toTensor();
      }
    }
  }

  template <typename TQueue>
  inline void convertOutput(const std::vector<::torch::IValue>& tensors,
                            TensorRegistry<TQueue>& outputs,
                            ::torch::Device device) {
    for (size_t i = 0; i < outputs.size(); i++) {
      // Only tensors are currenlty supported for conversion
      if (tensors[i].isTensor()) {
        arrayToTensor(device, outputs[i]) = tensors[i].toTensor();
      }
    }
  }

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAConversion_h
