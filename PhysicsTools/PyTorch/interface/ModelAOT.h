#ifndef PhysicsTools_PyTorch_interface_ModelAOT
#define PhysicsTools_PyTorch_interface_ModelAOT

#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

namespace cms::torch {

  using AotPkgLoader = ::torch::inductor::AOTIModelPackageLoader;

  // Note that this solution is in the beta state not stable for production use (experimental only).
  //
  // wrapper of AOTIModelPackageLoader
  // Following torchlib APIs are subject to change due to active development.
  // Authors provide NO BC guarantee for these APIs. Implementation based on 2.6 version.
  // Backward compatibility may be required to support multiple PyTorch versions within CMSSW.
  // See: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_package/model_package_loader.h#L8
  class ModelAOT {
  public:
    // Does not support async loading, the H2D copy is done on pageable memory in default torch stream.
    // Refer: PhysicsTools/PyTorch/test/testModelWrapperAot.cc -> testAsyncExecutionImplicitStream() / testAsyncExecutionExplicitStream()

    explicit ModelAOT(const std::string &precompiled_lib_path)
        : pkg_loader_(precompiled_lib_path), device_(::torch::Device(pkg_loader_.get_metadata()["AOTI_DEVICE_KEY"])) {}

    // Forward pass (inference) of model, returns std::vector<at::Tensor> (multi output support). Thread safety not verified yet.
    // Match native torchlib interface. cudaStream_t can be passed to run inference on specific stream.
    // If not passed then the one associated with device is grabed from thread local stream registry.
    // See: https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDAStream.cpp#L169
    //
    // Following torchlib APIs are subject to change due to active development.
    // Authors provide NO BC guarantee for these APIs. Implementation based on 2.6 version.
    // Backward compatibility may be required to support multiple PyTorch versions within CMSSW
    // See: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h#L9
    std::vector<at::Tensor> forward(std::vector<at::Tensor> &inputs, void *stream_handle = nullptr) {
      return pkg_loader_.run(inputs, stream_handle);
    }

    // Get model current device information.
    // TODO: verify if device can be changed after loading the model without reinitialization.
    //   Seems issues are resolved and model can be instantiated on multiple devices
    //   Refering to:
    //    - https://github.com/pytorch/pytorch/issues/136369
    //    - https://github.com/pytorch/pytorch/issues/141042
    //    - https://github.com/pytorch/pytorch/pull/136715
    ::torch::Device device() const { return device_; }

  protected:
    AotPkgLoader pkg_loader_;  // AOT package wrapper
    ::torch::Device device_;   // device
  };

}  // namespace cms::torch

#endif  // PhysicsTools_PyTorch_interface_ModelAOT
