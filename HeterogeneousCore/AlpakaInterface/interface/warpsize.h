#ifndef HeterogeneousCore_AlpakaInterface_interface_warpsize_h
#define HeterogeneousCore_AlpakaInterface_interface_warpsize_h

namespace cms::alpakatools {

  // TODO implement constexpt warp size in alpaka, and replace this workaround with that functionality.
#if defined(__SYCL_DEVICE_ONLY__)
// the warp size is not defined at compile time for SYCL backend
#error "The SYCL backend does not support compile-time warp size"
  inline constexpr int warpSize = 0;
#elif defined(__CUDA_ARCH__)
  // CUDA always has a warp size of 32
  inline constexpr int warpSize = 32;
#elif defined(__HIP_DEVICE_COMPILE__)
  // HIP/ROCm defines warpSize as a constant expression in device code, with value 32 or 64 depending on the target device
  inline constexpr int warpSize = ::warpSize;
#else
  // CPU back-ends always have a warp size of 1
  inline constexpr int warpSize = 1;
#endif

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_warpsize_h
