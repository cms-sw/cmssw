#ifndef PhysicsTools_PyTorch_interface_alpaka_Config_h
#define PhysicsTools_PyTorch_interface_alpaka_Config_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/TorchCompat.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torch {

  constexpr auto kDevHost = c10::DeviceType::CPU;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr auto kDevice = c10::DeviceType::CUDA;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  // AMD ROCm/HIP backend is not yet supported, fallback to CPU inference on these nodes.
  // See: https://github.com/pytorch/pytorch/blob/v2.6.0/aten/CMakeLists.txt#L73-L76
  // cms-sw/cmsdist PRs: https://github.com/cms-sw/cmsdist/pulls?q=is%3Apr+is%3Aopen+in%3Atitle+%22PyTorch%22
  constexpr auto kDevice = c10::DeviceType::CPU;
  // constexpr auto kDevice = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr auto kDevice = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr auto kDevice = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif

  inline ::torch::Device getDevice(const Device &device) {
    return (kDevice == kDevHost) ? ::torch::Device(kDevHost) : ::torch::Device(kDevice, device.getNativeHandle());
  }

  inline ::torch::Device getDevice(const Queue &queue) { return getDevice(::alpaka::getDev(queue)); }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torch

#endif  // PhysicsTools_PyTorch_interface_alpaka_Config_h