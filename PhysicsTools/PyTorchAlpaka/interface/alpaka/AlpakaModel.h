#ifndef PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModel_h
#define PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModel_h

#include "alpaka/alpaka.hpp"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Converter.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/Config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  using namespace cms::torch::alpakatools;

  class AlpakaModel : public cms::torch::Model {
  public:
    // inherit generic pytorch interface methods
    using cms::torch::Model::forward;
    using cms::torch::Model::to;

    // Default model loads to CPU memory space. Prefered way to load model,
    // Move to device memory space is done asynchronously using to() method in CMSSW aware stream.
    explicit AlpakaModel(const std::string &model_path) : cms::torch::Model(model_path) {}

    // Loads model to alpaka accelerator specified memory space.
    // Note that this is done in default stream, i.e. synchronously.
    explicit AlpakaModel(const std::string &model_path, const Device &dev)
        : cms::torch::Model(model_path, getDevice(dev)) {}
    explicit AlpakaModel(const std::string &model_path, const Queue &queue)
        : cms::torch::Model(model_path, getDevice(queue)) {}

    // Forward pass (inference) of model with SoA metadata input/output.
    // Allows to run inference directly using SoA portable objects/collections without excessive copies and conversions.
    // Refer: PhysicsTools/PyTorch/interface/Converter.h for details about wrapping memory layouts.
    void forward(Queue &queue, ModelMetadata &metadata) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      metadata.copyToHost(queue);
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      auto input_tensor = Converter::convert_input(metadata, device_);
      if (metadata.multi_head) {
        auto output_tensors = model_.forward(input_tensor);
        Converter::convert_output(output_tensors, metadata, device_);
      } else {
        Converter::convert_output(metadata, device_) = model_.forward(input_tensor).toTensor();
      }
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      metadata.copyToDevice(queue);
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    }

    // Move model to specified device memory space. Async load (in default stream if not overridden by the caller)
    // The caller should ensure the QueueGuard is instantiated and PyTorch stream context is properly set.
    void to(const Device &dev) {
      if constexpr (std::is_same_v<::alpaka::Dev<Device>, ::alpaka::DevCpu>) {
        this->Model::to(getDevice(dev));
        return;
      }
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      // ROCm/HIP not yet directly supported → fallback to CPU inference
      this->Model::to(getDevice(dev));
      return;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      // CUDA → keep async execution
      this->Model::to(getDevice(dev), true);
    }

    // Overload for Queue to simplify the interface for the common case of async execution.
    void to(const Queue &queue) { this->AlpakaModel::to(::alpaka::getDev(queue)); }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorchAlpaka_interface_alpaka_AlpakaModel_h