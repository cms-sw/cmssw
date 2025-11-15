#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <type_traits>

namespace cms::alpakatools {
  namespace warp {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using warp_mask_t = unsigned;  // 32-bit masks on NVIDIA
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using warp_mask_t = unsigned long;  // 64-bit masks on AMD (check!)
#endif

#else
    using warp_mask_t = unsigned;  // for host (no-op)
#endif

#ifdef __HIP_DEVICE_COMPILE__

#if HIP_VERSION_MAJOR < 7
#warning "HIP Version < 7.0 is not supported."
#endif

#endif

    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void syncWarpThreads_mask(TAcc const& acc, warp::warp_mask_t mask) {
      if (mask == 0)
        return;  //early return for the trivial mask

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      // Alpaka CUDA/HIP backend
      __syncwarp(mask);  // Synchronize all threads within a subset of lanes in the warp
#endif
      // No-op for CPU accelerators
    }

    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE warp::warp_mask_t ballot_mask(TAcc const& acc,
                                                                      warp::warp_mask_t mask,
                                                                      int pred) {
      warp::warp_mask_t res{0};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __ballot_sync(mask, pred);
#else
      res = pred == 1 ? mask : 0;
#endif
      return res;
    }

    template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T
    shfl_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_sync(mask, var, srcLane, width);  // Synchronize all threads within a warp
#else
      res = var;
#endif

      return res;
    }

    template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T
    shfl_down_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_down_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T
    shfl_up_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_up_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE warp::warp_mask_t match_any_mask(TAcc const& acc,
                                                                         warp::warp_mask_t mask,
                                                                         T val) {
      warp::warp_mask_t res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#if __CUDA_ARCH__ >= 700 || ALPAKA_ACC_GPU_HIP_ENABLED
      res = __match_any_sync(mask, val);
#else
      const unsigned int w_extent = alpaka::warp::getSize(acc);
      unsigned int match = 0;
      for (int iter_lane_idx = 0; iter_lane_idx < w_extent; ++iter_lane_idx) {
        T iter_val = __shfl_sync(mask, val, iter_lane_idx, w_extent);
        const unsigned int iter_lane_mask = 1 << iter_lane_idx;
        if (iter_val == val)
          match |= iter_lane_mask;
      }
      res = match & mask;
#endif
#else
      res = mask;
#endif
      return res;
    }

  }  // namespace warp

  // reverse the bit order of a (32-bit) unsigned integer.
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE warp::warp_mask_t brev(TAcc const& acc, warp::warp_mask_t mask) {
    warp::warp_mask_t res{0};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // Alpaka CUDA backend
    res = __brev(mask);
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // Alpaka HIP backend
    res = __brevll(mask);
#endif
#else
    res = mask;
#endif
    return res;
  }

  // count the number of leading zeros in a 32-bit unsigned integer
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned clz(TAcc const& acc, warp::warp_mask_t mask) {
    unsigned res{0};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // Alpaka CUDA backend
    res = __clz(mask);
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // Alpaka HIP backend
    res = __clzll(mask);
#endif
#else
    res = mask == 0 ? 1 : 0;
#endif
    return res;
  }

}  // namespace cms::alpakatools
#endif
