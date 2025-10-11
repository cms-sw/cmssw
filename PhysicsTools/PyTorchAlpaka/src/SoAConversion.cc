#include "PhysicsTools/PyTorchAlpaka/interface/SoAConversion.h"

#include <cassert>

namespace cms::torch::alpakatools {

  // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
  std::vector<::torch::IValue> convertInput(const TensorRegistry& inputs, ::torch::Device device) {
    std::vector<::torch::IValue> tensors(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      assert(reinterpret_cast<intptr_t>(inputs[i].data()) % inputs[i].alignment() == 0);
      tensors[i] = arrayToTensor(device, inputs[i]);
    }
    return tensors;
  }

  // Calculate size and stride of data store based on OutputMetadata and fill SoA with tensor values
  // TODO: temporary solution for multi output branch models, figure out how to solve without copy (similar issue to AOT compiled models)
  void convertOutput(const ::torch::IValue& tensors, const TensorRegistry& outputs, ::torch::Device device) {
    if (tensors.isTuple()) {
      const auto tensors_tuple = tensors.toTuple();
      for (size_t i = 0; i < outputs.size(); i++) {
        assert(reinterpret_cast<intptr_t>(outputs[i].data()) % outputs[i].alignment() == 0);
        arrayToTensor(device, outputs[i]) = tensors_tuple->elements()[i].toTensor();
      }
    }
  }

  // Calculate size and stride of data store based on OutputMetadata and fill SoA with tensor values
  void convertOutput(const std::vector<::torch::IValue>& tensors,
                     const TensorRegistry& outputs,
                     ::torch::Device device) {
    for (size_t i = 0; i < outputs.size(); i++) {
      // Only tensors are currenlty supported for conversion
      if (tensors[i].isTensor()) {
        assert(reinterpret_cast<intptr_t>(outputs[i].data()) % outputs[i].alignment() == 0);
        arrayToTensor(device, outputs[i]) = tensors[i].toTensor();
      }
    }
  }

  ::torch::Tensor convertOutput(const TensorRegistry& outputs, ::torch::Device device) {
    assert(reinterpret_cast<intptr_t>(outputs[0].data()) % outputs[0].alignment() == 0);
    return arrayToTensor(device, outputs[0]);
  }

  // Wrap raw pointer by torch::Tensor based on type, size and stride.
  ::torch::Tensor arrayToTensor(::torch::Device device, const PortableTensorHandle& tensor_handle) {
    // const_cast is required as `from_blob` does not take const pointer even if it does not modify the data
    // see: https://discuss.pytorch.org/t/using-torch-from-blob-with-const-data/141597
    // https://github.com/pytorch/pytorch/blob/89a6dbe73af4ca64ee26f4e46219e163b827e698/aten/src/ATen/ops/from_blob.h#L107
    //
    // TODO: open issue to `pytorch` repo:
    //  - see if they can add const correctness, or get to know why const is currently prevented?
    auto options = ::torch::TensorOptions().dtype(tensor_handle.type()).device(device).pinned_memory(true);
    return ::torch::from_blob(
        const_cast<void*>(tensor_handle.data()), tensor_handle.sizes(), tensor_handle.strides(), options);
  }

  // // AOT specific implementation, as model expects vector of torch::Tensor not torch::IValue
  // std::vector<::torch::Tensor> convertInputTensor(const ModelMetadata& metadata, ::torch::Device device) {
  //   std::vector<::torch::Tensor> tensors(metadata.input.nBlocks);
  //   for (int i = 0; i < metadata.input.nBlocks; i++) {
  //     assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr()) %
  //                 metadata.input[metadata.input.order[i]].alignment() ==
  //             0);
  //     tensors[i] = arrayToTensor(device, metadata.input[metadata.input.order[i]]);
  //   }
  //   return tensors;
  // }

  // AOT specific implementation, as return type is torch::Tensor not torch::IValue
  // void convertOutput(const std::vector<::torch::Tensor>& tensors,
  //                    const ModelMetadata& metadata,
  //                    ::torch::Device device) {
  //   for (int i = 0; i < metadata.output.nBlocks; i++) {
  //     assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[i]].ptr()) %
  //                 metadata.output[metadata.output.order[i]].alignment() ==
  //             0);
  //     arrayToTensor(device, metadata.output[metadata.output.order[i]]) = tensors[i];
  //   }
  // }

}  // namespace cms::torch::alpakatools
