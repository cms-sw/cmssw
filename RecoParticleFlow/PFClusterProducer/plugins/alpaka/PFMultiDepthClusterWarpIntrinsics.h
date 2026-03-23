#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterWarpIntrinsics_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <type_traits>
#include <concepts>

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

#ifdef __HIP_DEVICE_COMPILE__

#if !((HIP_VERSION_MAJOR >= 7) || (HIP_VERSION >= 60200000 && defined(HIP_ENABLE_WARP_SYNC_BUILTINS)))
#warning "HIP Version is not supported."
#endif

#endif

    /**
 * @brief Synchronize all threads within a subset of lanes in the warp
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask Input mask.
 */

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void syncWarpThreads_mask(TAcc const& acc, warp::warp_mask_t mask) {
      if (mask == 0)
        return;  //early return for the trivial mask

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      // Alpaka CUDA/HIP backend
      __syncwarp(mask);  // Synchronize all threads within a subset of lanes in the warp
#endif
      // No-op for CPU accelerators
    }

    /**
 * @brief Warp-wide ballot of a predicate, restricted to a given active-lane mask.
 *
 * Computes a warp mask containing the lanes for which 'pred' is non-zero,
 * considering only lanes enabled in 'mask'. 
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Active-lane mask defining which lanes participate in the ballot.
 * @param pred Per-lane predicate value; non-zero counts as 'true'.
 *
 * @return A warp mask with bits set for participating lanes (as defined by 'mask')
 *         whose 'pred' evaluates to 'true'.
 */
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE warp::warp_mask_t ballot_mask(TAcc const& acc, warp::warp_mask_t mask, int pred) {
      warp::warp_mask_t res{0};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __ballot_sync(mask, pred);
#else
      res = pred == 0 ? 0 : mask;
#endif
      return res;
    }
    /**
 * @brief Masked warp shuffle from a source lane.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Source lane index within the shuffle width.
 * @param width   Logical warp width for the shuffle. 
 *
 * @return The value of 'var' from lane 'srcLane', or an unspecified value if
 *         the source lane is inactive (i.e., not set in 'mask' ).
 */
    template <alpaka::concepts::Acc TAcc, typename T>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE T shfl_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_sync(mask, var, srcLane, width);  // Synchronize all threads within a warp
#else
      res = var;
#endif

      return res;
    }

    /**
 * @brief Masked warp shuffle-down operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Lane offset (delta) below the calling lane.
 * @param width   Logical warp width for the shuffle.
 *
 * @return The value of 'var' from the source lane, or an unspecified value if
 *         the source lane is inactive or out of range.
 */

    template <alpaka::concepts::Acc TAcc, typename T>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE T
    shfl_down_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_down_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    /**
 * @brief Masked warp shuffle-up operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type to be shuffled.
 *
 * @param acc     Alpaka accelerator instance.
 * @param mask    Active-lane mask for the shuffle operation.
 * @param var     Per-lane input value.
 * @param srcLane Lane offset (delta) above the calling lane.
 * @param width   Logical warp width for the shuffle.
 *
 * @return The value of 'var' from the source lane, or an unspecified value if
 *         the source lane is inactive or out of range.
 */

    template <alpaka::concepts::Acc TAcc, typename T>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE T
    shfl_up_mask(TAcc const& acc, warp::warp_mask_t mask, T var, int srcLane, int width) {
      T res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      res = __shfl_up_sync(mask, var, srcLane, width);
#else
      res = var;
#endif
      return res;
    }

    /**
 * @brief Masked warp-wide match-any operation.
 *
 * @tparam TAcc Alpaka accelerator type. 
 * @tparam T    Value type used for the match comparison.
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Active-lane mask for the match operation.
 * @param val  Per-lane value to be compared across the warp.
 *
 * @return A warp mask with bits set for lanes (enabled in 'mask') whose
 *         'val' equals the calling lane’s value.
 */
    template <alpaka::concepts::Acc TAcc, typename T>
      requires std::is_arithmetic_v<T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE warp::warp_mask_t match_any_mask(TAcc const& acc, warp::warp_mask_t mask, T val) {
      warp::warp_mask_t res{};
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#if __CUDA_ARCH__ >= 700 || ALPAKA_ACC_GPU_HIP_ENABLED
      res = __match_any_sync(mask, val);
#else
      constexpr unsigned int w_extent = alpaka::warp::getSizeCompileTime<TAcc>();
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

  /**
 * @brief Reverse the bit order of a warp mask.
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Input warp mask whose bit order is to be reversed.
 *
 * @return A warp mask with 32/64 bits reversed.
 */

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE warp::warp_mask_t brev(TAcc const& acc, warp::warp_mask_t mask) {
    warp::warp_mask_t res{0};
#if defined(__CUDA_ARCH__)
    // Alpaka CUDA backend
    res = __brev(mask);
#elif defined(__HIP_DEVICE_COMPILE__)
    // Alpaka HIP backend
    res = __brevll(mask);
#else
    res = mask;
#endif
    return res;
  }

  /**
 * @brief Count leading zeros in a warp mask.
 *
 * @tparam TAcc Alpaka accelerator type. 
 *
 * @param acc  Alpaka accelerator instance.
 * @param mask Input warp mask.
 *
 * @return The number of leading zero bits in the lower 32/64 bits of 'mask'.
 */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned clz(TAcc const& acc, warp::warp_mask_t mask) {
    unsigned res{0};
#if defined(__CUDA_ARCH__)
    // Alpaka CUDA backend
    res = __clz(mask);
#elif defined(__HIP_DEVICE_COMPILE__)
    // Alpaka HIP backend
    res = __clzll(mask);
#else
    res = mask == 0 ? 1 : 0;
#endif
    return res;
  }

}  // namespace cms::alpakaintrinsics
#endif
