#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerHelper_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerHelper_h

/**
 * @file PFMultiDepthClusterizerHelper.h
 * @brief Warp-level utility functions for particle flow multi-depth clustering.
 * 
 * This header provides basic warp-synchronous operations used in clustering algorithms,
 * including bitwise manipulations (least/most significant set bits) and masked
 * warp-exclusive sum computations.
 */

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"

#include <concepts>
#include <type_traits>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace cms::alpakaintrinsics;

  /**
 * @brief Compute warp size
 *
 * @param mask Input lane index in the warp
 * 
 * @return compute lane mask:
 */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr std::int32_t get_warp_size() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return alpaka::warp::getSizeCompileTime<TAcc>();
#else
    return 1;
#endif
  }

  /**
 * @brief Compute lane mask
 *
 * @param mask Input lane index in the warp
 * 
 * @return compute lane mask:
 */
  ALPAKA_FN_ACC constexpr inline warp::warp_mask_t get_lane_mask(const unsigned int lane_idx) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return (1ULL << lane_idx);
#else
    return (1U << lane_idx);
#endif
  }

  /**
 * @brief Check that given lane is active in the custom lane mask
 *
 * @param mask Input mask.
 * @param mask Input lane index in the warp
 * @param w_extent (Sub)warp size
 * 
 * @return True if active, otherwise false.
 */
  ALPAKA_FN_ACC constexpr inline bool is_work_lane(const warp::warp_mask_t work_mask,
                                                   const unsigned int lane_idx,
                                                   const unsigned extent) {
    if (lane_idx >= extent)
      return false;
    return ((work_mask >> lane_idx) & 1);
  }

  /**
 * @brief Returns the position of the least significant set bit in a mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  Input mask.
 * 
 * @return Index of least significant 1 bit (0-based). (or warp size if x == 0).
 */
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE warp::warp_mask_t get_ls1b_idx(TAcc const& acc, const warp::warp_mask_t mask) {
    if (mask == 0)
      return static_cast<warp::warp_mask_t>(get_warp_size<TAcc>());

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return 0;

    using signed_warp_mask_t = std::make_signed_t<warp::warp_mask_t>;

    const auto pos = alpaka::ffs(acc, static_cast<signed_warp_mask_t>(mask));
    return static_cast<warp::warp_mask_t>(pos - 1);
  }

  /**
 * @brief Performs warp-level exclusive prefix sum
 *
 * @tparam TAcc Alpaka accelerator type.
 * @tparam accum If true, broadcast total accumulated value to lowest active lane.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param val   Value to include in the prefix sum.
 * @param lane_idx Current thread's lane index.
 * 
 * @return Exclusive prefix sum value for the current lane.
 * convention used here:
 * - lanes 1..(w_extent-1) receive the exclusive prefix sum (CSR offsets within the warp),
 * - lane 0 receives the total sum over the warp (used as the per-warp NNZ aggregate)
 */

  template <alpaka::concepts::Acc TAcc, bool all = true>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int warp_exclusive_sum(TAcc const& acc,
                                                                 unsigned int val,
                                                                 const unsigned int lane_idx) {
    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return all ? val : 0;

    const auto full_mask = alpaka::warp::ballot(acc, true);

    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    unsigned int local_offset = val;

    // Do inclusive sum first:
    //CMS_UNROLL_LOOP
    for (unsigned int step = 1; step < w_extent; step *= 2) {
      const auto res = alpaka::warp::shfl_up(acc, local_offset, step, w_extent);
      if (lane_idx >= step)
        local_offset += res;
    }

    warp::syncWarpThreads_mask(acc, full_mask);

    if constexpr (all) {
      const unsigned int high_lane_idx = w_extent - 1;

      // send last lane value (total tile offset) to lane idx = low_lane_idx:
      const warp::warp_mask_t active_mask = 1 | get_lane_mask(high_lane_idx);
      const unsigned int tmp = warp::shfl_mask(acc, active_mask, local_offset, high_lane_idx, w_extent);

      if (lane_idx == 0)
        local_offset = tmp;  //lane 0 keeps full (inclusive for the last lane) sum
    }
    return lane_idx == 0 ? local_offset : local_offset - val;  //we return exclusive sum!
  }

  /**
 * @brief Returns logical index for a given physical lane index based on custom lane mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param mask Input bitmask.
 * @param lane_idx imput phys. lane index
 * 
 * @return Index of the lane in the mask 
 */

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int get_logical_lane_idx(TAcc const& acc,
                                                                   const warp::warp_mask_t mask,
                                                                   const unsigned int lane_idx) {
    if (lane_idx == 0)
      return lane_idx;  // nothing to do, phys idx coincide with the logical one.
    const warp::warp_mask_t lane_mask = mask & (get_lane_mask(lane_idx) - 1);
    return alpaka::popcount(acc, lane_mask);  // Count 1s below current lane
  }

  /**
 * @brief Returns physical lane index for a given logical lane index based on custom lane mask.
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param mask Input mask.
 * @param logical_lane_idx input logical lane index
 * 
 * @return Physical index of the lane in the mask 
 */

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int get_physical_lane_idx(TAcc const& acc,
                                                                    const warp::warp_mask_t mask,
                                                                    int logical_lane_idx) {
    using signed_warp_mask_t = std::make_signed_t<warp::warp_mask_t>;

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return 0;

    signed_warp_mask_t m = mask;

    while (logical_lane_idx--)
      m &= (m - 1);

    const auto pos = alpaka::ffs(acc, m);

    return static_cast<unsigned int>(pos - 1);
  }

  /**
 * @brief generic warp reduction
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance.
 * @param in input value to reduce
 * @param f reducer 
 * 
 * @return return reduced value (propagated to all lanes in the mask by default)
 */

  template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
    requires std::is_arithmetic_v<reduce_t>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE reduce_t warp_reduce(TAcc const& acc, reduce_t const in, const reducer_t f) {
    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    reduce_t result = in;

    if constexpr (std::is_same_v<Device, alpaka::DevCpu>)
      return result;

    for (unsigned int offset = w_extent / 2; offset > 0; offset /= 2) {
      result = f(result, alpaka::warp::shfl_down(acc, result, offset, w_extent));
    }

    if constexpr (all)
      result = alpaka::warp::shfl(acc, result, 0, w_extent);

    return result;
  }

  /**
 * @brief Sparse warp reduction
 *
 * @tparam TAcc Alpaka accelerator type.
 * 
 * @param acc Alpaka accelerator instance
 * @param mask input mask 
 * @param in input value to reduce
 * @param f reducer 
 * 
 * @return return reduced value (propagated to all lanes in the mask by default)
 */

  template <alpaka::concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
    requires std::is_arithmetic_v<reduce_t>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE reduce_t warp_sparse_reduce(TAcc const& acc,
                                                             const warp::warp_mask_t mask,
                                                             const unsigned int lane_idx,
                                                             reduce_t const in,
                                                             const reducer_t f) {
    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return mask == 0 ? 0 : in;

    // Non-active lanes should skip the reduction:
    if (is_work_lane(mask, lane_idx, w_extent) == false)
      return in;

    unsigned int nActiveLanes = alpaka::popcount(acc, mask);  // count number of active lanes

    // First check if this is just a single active lane in the warp:
    if (nActiveLanes == 1)
      return in;

    //Compute the next power of two:
    const unsigned int pow2 = w_extent - cms::alpakaintrinsics::clz(acc, nActiveLanes - 1);
    const unsigned int pow2_boundary = 1 << pow2;

    const unsigned int logical_lane_idx = get_logical_lane_idx(acc, mask, lane_idx);

    reduce_t res = in;

    for (unsigned int offset = pow2_boundary / 2; offset > 0; offset /= 2) {
      const unsigned int logical_src_lane_idx = logical_lane_idx + offset;
      const unsigned int src_lane_idx =
          (logical_src_lane_idx < nActiveLanes) ? get_physical_lane_idx(acc, mask, logical_src_lane_idx) : lane_idx;

      const reduce_t neigh_res = warp::shfl_mask(acc, mask, res, src_lane_idx, w_extent);

      if (logical_src_lane_idx < nActiveLanes)
        res = f(res, neigh_res);

      warp::syncWarpThreads_mask(acc, mask);
    }

    if constexpr (all) {
      const auto low_lane_idx = get_physical_lane_idx(acc, mask, 0);
      res = warp::shfl_mask(acc, mask, res, low_lane_idx, w_extent);
    }

    return res;
  }

  /**
 * @brief Performs warp-level sparse exclusive prefix sum (masked version of warp_exclusive_sum, see above )
 *
 * @tparam TAcc Alpaka accelerator type.
 * @tparam accum If true, broadcast total accumulated value to lowest active lane.
 * 
 * @param acc   Alpaka accelerator instance.
 * @param mask  input mask 
 * @param val   Value to include in the prefix sum.
 * @param lane_idx Current thread's lane index.
 * 
 * @return Exclusive prefix sum value for the current lane.
 */

  template <alpaka::concepts::Acc TAcc, bool all = true>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int warp_sparse_exclusive_sum(TAcc const& acc,
                                                                        const warp::warp_mask_t mask,
                                                                        const unsigned int val,
                                                                        const unsigned int lane_idx) {
    constexpr unsigned int w_extent = get_warp_size<TAcc>();

    if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
      return all == false ? 0 : (mask == 0 ? 0 : val);

    // Non-active lanes should skip the reduction:
    if (is_work_lane(mask, lane_idx, w_extent) == false)
      return 0;

    // count number of active lanes
    const unsigned int nActiveLanes = alpaka::popcount(acc, mask);
    // First check if this is just a single active lane in the warp:
    if (nActiveLanes == 1)
      return val;  //nothing to do, note that this is the inclusive "sum": low lane always keeps the whole sum

    //Compute the next power of two:
    const unsigned int pow2 = w_extent - cms::alpakaintrinsics::clz(acc, nActiveLanes - 1);
    const unsigned int pow2_boundary = 1 << pow2;

    const unsigned int logical_lane_idx = get_logical_lane_idx(acc, mask, lane_idx);

    unsigned int local_offset = val;

    for (unsigned int step = 1; step < pow2_boundary; step *= 2) {
      const unsigned int src_lane_idx =
          (logical_lane_idx >= step) ? get_physical_lane_idx(acc, mask, logical_lane_idx - step) : lane_idx;
      const unsigned int tmp_val = warp::shfl_mask(acc, mask, local_offset, src_lane_idx, w_extent);

      if (logical_lane_idx >= step)
        local_offset += tmp_val;
    }

    if constexpr (all) {
      const unsigned int high_lane_idx = get_physical_lane_idx(acc, mask, nActiveLanes - 1);
      // send last lane value (total tile offset) to lane idx = low_lane_idx:
      const unsigned int tmp = warp::shfl_mask(acc, mask, local_offset, high_lane_idx, w_extent);

      if (logical_lane_idx == 0)
        local_offset = tmp;  //lane 0 keeps full (inclusive for the last lane) sum
    }
    return logical_lane_idx == 0
               ? local_offset
               : local_offset - val;  //we return exclusive sum, except zero logical lane (which returns total offset)
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
