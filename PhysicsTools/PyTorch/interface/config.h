#ifndef PhysicsTools_PyTorch_interface_config_h
#define PhysicsTools_PyTorch_interface_config_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace torch_common {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CUDA;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;
#else
  #error "Could not define the torch device type."
#endif  
}

#endif // defined PhysicsTools_PyTorch_interface_config_h