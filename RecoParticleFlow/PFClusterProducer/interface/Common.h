#ifndef PFClusterProducer_interface_Common_h
#define PFClusterProducer_interface_Common_h

#include <cstdint>

namespace cms::alpakaintrinsics {
  namespace warp {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using warp_mask_t = std::uint32_t;  // 32-bit masks on NVIDIA
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using warp_mask_t = std::uint64_t;  // 64-bit masks on AMD (!)
#endif

#else
    using warp_mask_t = std::uint32_t;  // for host (no-op)
#endif

  }  // namespace warp
}  // namespace cms::alpakaintrinsics

struct ccSeed {
  uint32_t value;
  uint32_t key;
};

#endif
