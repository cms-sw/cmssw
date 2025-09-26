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
  // HIP/ROCm may have a warp size of 32 or 64 depending on the target device
#if defined(__gfx900__) or defined(__gfx902__) or defined(__gfx903__) or defined(__gfx906__) or defined(__gfx908__) or \
    defined(__gfx909__) or defined(__gfx90a__) or defined(__gfx90c__) or defined(__gfx942__) or defined(__gfx950__)
  inline constexpr int warpSize = 64;
#elif defined(__gfx1010__) or defined(__gfx1011__) or defined(__gfx1012__) or defined(__gfx1013__) or \
    defined(__gfx1030__) or defined(__gfx1031__) or defined(__gfx1032__) or defined(__gfx1033__) or   \
    defined(__gfx1034__) or defined(__gfx1035__) or defined(__gfx1036__) or defined(__gfx1100__) or   \
    defined(__gfx1101__) or defined(__gfx1102__) or defined(__gfx1103__) or defined(__gfx1150__) or   \
    defined(__gfx1151__) or defined(__gfx1152__) or defined(__gfx1153__) or defined(__gfx1200__) or   \
    defined(__gfx1201__) or defined(__gfx1250__) or defined(__gfx1251__)
  inline constexpr int warpSize = 32;
#else
#error "Unknown AMDGCN architecture"
#endif
#else
  // CPU back-ends always have a warp size of 1
  inline constexpr int warpSize = 1;
#endif

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_warpsize_h
