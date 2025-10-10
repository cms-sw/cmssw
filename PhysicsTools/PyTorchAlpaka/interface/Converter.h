#ifndef PhysicsTools_PyTorchAlpaka_interface_Converter_h
#define PhysicsTools_PyTorchAlpaka_interface_Converter_h

#include <cassert>
#include <cstdint>
#include <vector>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "PhysicsTools/PyTorchAlpaka/interface/SoAMetadata.h"

namespace cms::torch::alpakatools {

  // Static class to wrap raw SOA pointer in tensor object without copying.
  class Converter {
  public:
    // Calculate size and stride of data store based on InputMetadata and return list of IValue, which is parent class of torch::tensor.
    static std::vector<::torch::IValue> convert_input(const ModelMetadata& metadata, ::torch::Device device) {
      std::vector<::torch::IValue> tensors(metadata.input.nBlocks);
      for (int i = 0; i < metadata.input.nBlocks; i++) {
        assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr()) %
                   metadata.input[metadata.input.order[i]].alignment() ==
               0);
        tensors[i] = Converter::array_to_tensor(device, metadata.input[metadata.input.order[i]]);
      }
      return tensors;
    }

    // AOT specific implementation, as model expects vector of torch::Tensor not torch::IValue
    static std::vector<::torch::Tensor> convert_input_tensor(const ModelMetadata& metadata, ::torch::Device device) {
      std::vector<::torch::Tensor> tensors(metadata.input.nBlocks);
      for (int i = 0; i < metadata.input.nBlocks; i++) {
        assert(reinterpret_cast<intptr_t>(metadata.input[metadata.input.order[i]].ptr()) %
                   metadata.input[metadata.input.order[i]].alignment() ==
               0);
        tensors[i] = Converter::array_to_tensor(device, metadata.input[metadata.input.order[i]]);
      }
      return tensors;
    }

    // Calculate size and stride of data store based on OutputMetadata and return single output tensor
    static ::torch::Tensor convert_output(const ModelMetadata& metadata, ::torch::Device device) {
      assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[0]].ptr()) %
                 metadata.output[metadata.output.order[0]].alignment() ==
             0);
      return Converter::array_to_tensor(device, metadata.output[metadata.output.order[0]]);
    }

    // Calculate size and stride of data store based on OutputMetadata and fill SoA with tensor values
    // TODO: temporary solution for multi output branch models, figure out how to solve without copy (similar issue to AOT compiled models)
    static void convert_output(const ::torch::IValue& tensors, const ModelMetadata& metadata, ::torch::Device device) {
      if (tensors.isTuple()) {
        const auto tensors_tuple = tensors.toTuple();
        for (int i = 0; i < metadata.output.nBlocks; i++) {
          assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[i]].ptr()) %
                     metadata.output[metadata.output.order[i]].alignment() ==
                 0);
          Converter::array_to_tensor(device, metadata.output[metadata.output.order[i]]) =
              tensors_tuple->elements()[i].toTensor();
        }
      }
    }

    // Calculate size and stride of data store based on OutputMetadata and fill SoA with tensor values
    static void convert_output(const std::vector<::torch::IValue>& tensors,
                               const ModelMetadata& metadata,
                               ::torch::Device device) {
      for (int i = 0; i < metadata.output.nBlocks; i++) {
        // Only tensors are currenlty supported for conversion
        if (tensors[i].isTensor()) {
          assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[i]].ptr()) %
                     metadata.output[metadata.output.order[i]].alignment() ==
                 0);
          Converter::array_to_tensor(device, metadata.output[metadata.output.order[i]]) = tensors[i].toTensor();
        }
      }
    }

    // AOT specific implementation, as return type is torch::Tensor not torch::IValue
    static void convert_output(const std::vector<::torch::Tensor>& tensors,
                               const ModelMetadata& metadata,
                               ::torch::Device device) {
      for (int i = 0; i < metadata.output.nBlocks; i++) {
        assert(reinterpret_cast<intptr_t>(metadata.output[metadata.output.order[i]].ptr()) %
                   metadata.output[metadata.output.order[i]].alignment() ==
               0);
        Converter::array_to_tensor(device, metadata.output[metadata.output.order[i]]) = tensors[i];
      }
    }

  private:
    // Wrap raw pointer by torch::Tensor based on type, size and stride.
    static ::torch::Tensor array_to_tensor(::torch::Device device, const Block& block) {
      auto options = ::torch::TensorOptions().dtype(block.type()).device(device).pinned_memory(true);
      // const_cast is required as `from_blob` does not take const pointer even if it does not modify the data
      // see: https://discuss.pytorch.org/t/using-torch-from-blob-with-const-data/141597
      // https://github.com/pytorch/pytorch/blob/89a6dbe73af4ca64ee26f4e46219e163b827e698/aten/src/ATen/ops/from_blob.h#L107
      //
      // TODO: open issue to `pytorch` repo:
      //  - see if they can add const correctness, or get to know why const is currently prevented?
      return ::torch::from_blob(const_cast<void*>(block.ptr()), block.size(), block.stride(), options);
    }
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_Converter_h