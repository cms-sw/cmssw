#ifndef PhysicsTools_PyTorchAlpaka_interface_GetDevice_h
#define PhysicsTools_PyTorchAlpaka_interface_GetDevice_h

#include <type_traits>

#include "alpaka/alpaka.hpp"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

namespace cms::torch::alpakatools {

  template <typename TDev>
    requires ::alpaka::concepts::Device<TDev>
  inline ::torch::Device getDevice(const TDev& device) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TDev, alpaka::DevCudaRt>)
      return ::torch::Device(c10::DeviceType::CUDA, device.getNativeHandle());
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    if constexpr (std::is_same_v<TDev, alpaka::DevHipRt>)
      return ::torch::Device(c10::DeviceType::CUDA, device.getNativeHandle());
#else
    // default, omit device index for CPU
    return ::torch::Device(c10::DeviceType::CPU);
#endif
  }

  template <typename TQueue>
    requires ::alpaka::concepts::Queue<TQueue>
  inline ::torch::Device getDevice(const TQueue& queue) {
    return getDevice(alpaka::getDev(queue));
  }

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_GetDevice_h
