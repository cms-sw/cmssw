#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthShowerShape_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthShowerShape_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include <cmath>
#include <limits>
#include <type_traits>
#include <concepts>

namespace cms::alpakamath {

  /**
 * @brief Return the scale factor used for longitudinal (z) coordinate normalization.
 *
 * Provides a compile-time constant scale factor used to normalize the z coordinate,
 * directly ported from the corresponding implementation in ROOT. The returned value
 * depends on the floating-point precision to preserve the numerical behavior of the
 * original algorithm.
 *
 * - For `float`, the value `53.81737057623773f` corresponds to the single precision
 *   scaling used in ROOT to map the detector z-range into a numerically stable domain.
 * - For `double`, the value `8192.0` is used, matching the double-precision scaling
 *   factor in ROOT and providing higher dynamic range.
 *
 * @tparam T Floating-point type (`float` or `double`).
 *
 * @return The z-coordinate scaling factor for type T.
 */
  // epsilon^{1/4}:
  template <std::floating_point T>
  constexpr T z_scaled() {
    using U = std::remove_cv_t<T>;

    if constexpr (std::is_same_v<U, float>) {
      constexpr float z_scaled_f = 53.81737057623773f;
      return z_scaled_f;
    } else {
      constexpr double z_scaled_d = 8192.0;
      return z_scaled_d;
    }
  }

  /**
 * @brief Compute the pseudorapidity from Cartesian coordinates
 *        using a ROOT-based implementation optimized for numerical stability.
 * 
 * @tparam TAcc Alpaka accelerator type. Enabled only if `alpaka::isAccelerator<TAcc>` is true.
 * @tparam T    Floating-point value type.
 *
 * @param acc Alpaka accelerator instance.
 * @param x   x coordinate.
 * @param y   y coordinate.
 * @param z   z coordinate.
 *
 * @return The pseudorapidity. For rho = sqrt{x^2 + y^2} = 0, returns
 *         +/-infinity depending on the sign of z.
 */

  template <alpaka::concepts::Acc TAcc, std::floating_point T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE T eta(TAcc const& acc, const T x, const T y, const T z) {
    // ROOT-style fast path:
    // uses log(zs + sqrt(zs^2 + 1)) when |zs| is moderate,
    // and a Taylor-aided form when |zs| is large.
    // For rho == 0, return +/-inf.

    T eta_val{0};

    const T rho = alpaka::math::sqrt(acc, x * x + y * y);

    if (rho > T{0}) {
      // threshold for switching to the Taylor-aided branch
      const T zscaled = z_scaled<T>();

      const T zs = z / rho;
      if (alpaka::math::abs(acc, zs) < zscaled) {
        eta_val = alpaka::math::log(acc, zs + alpaka::math::sqrt(acc, zs * zs + T{1}));
      } else {
        // first-order Taylor expansion for the sqrt part
        eta_val = (z > T{0}) ? alpaka::math::log(acc, T{2} * zs + T{0.5} / zs) : -alpaka::math::log(acc, -T{2} * zs);
      }
    } else {
      // Exactly along the beam axis:
      if (z != T{0})
        eta_val = alpaka::math::copysign(acc, std::numeric_limits<T>::infinity(), z);
    }
    return eta_val;
  }
}  // namespace cms::alpakamath

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace cms::alpakaintrinsics;

  /**
 * @brief Alpaka kernel computing shower-shape variables for PF clusters.
 *
 *
 * @tparam is_cooperative  Enable cooperative warp-level processing if 'true'.
 *                         the default value is 'false' to adopt current random access pattern. 
 * @param acc               Alpaka accelerator instance.
 * @param mdpfClusteringVars Output view for multi-depth clustering variables.
 * @param pfClusters        Input cluster device collection.
 * @param pfRecHitFracs     Input rechit fraction device collection.
 * @param pfRecHit          Input rechit device collection.
* @param rms2_threshold     Threshold applied to the RMS^2-based selection.
 */

  template <bool is_cooperative = false>
  class ShowerShapeKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const reco::PFClusterDeviceCollection::ConstView pfClusters,
                                  const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHit,
                                  const float rms2_threshold = 0.1) const {
      const unsigned int nClusters = pfClusters.nSeeds();

      if (::cms::alpakatools::once_per_grid(acc)) {
        mdpfClusteringVars.size() = nClusters;
      }

      const unsigned int w_extent = alpaka::warp::getSize(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int eff_w_extent = alpaka::popcount(acc, active_lanes_mask);
          // Skip inactive lanes:
          if (idx.global >= nClusters)
            continue;
          const int i = idx.global;

          const auto pfc_energy = pfClusters[i].energy();

          mdpfClusteringVars[i].depth() = pfClusters[i].depth();
          mdpfClusteringVars[i].energy() = pfc_energy;

          const double x_c = pfClusters[i].x();
          const double y_c = pfClusters[i].y();
          const double z_c = pfClusters[i].z();

          const float eta_c = static_cast<float>(cms::alpakamath::eta(acc, x_c, y_c, z_c));
          const float phi_c = static_cast<float>(::cms::alpakatools::phi(acc, x_c, y_c));

          mdpfClusteringVars[i].eta() = eta_c;
          mdpfClusteringVars[i].phi() = phi_c;

          int pfrhf_offset = pfClusters[i].rhfracOffset();
          int pfrhf_size = pfClusters[i].rhfracSize();

          auto addFn = [] ALPAKA_FN_ACC(double a, double b) -> double { return a + b; };

          // Cooperative mode: each "master" lane (iter_lane_idx) owns a PFRecHitFraction span
          // [pfrhf_offset, pfrhf_offset + pfrhf_size). If that span is longer than 1, we recruit a
          // subgroup of currently "free" lanes to help process the span in parallel.
          //
          // Mask conventions in this scope:
          // - active_lanes_mask : lanes corresponding to in-range clusters (plus any forced lanes).
          // - free_lanes_mask   : lanes currently available to be assigned as cooperative workers
          //                       (subset of active_lanes_mask).
          //                       if a swap_lanes_mask (see below) is a non-empty set, then free lanes mask
          //                       also includes lanes for later swap operation
          // - swap_lanes_mask   : lanes that currently hold state and must be preserved to be swapped out to some free lanes.
          if constexpr (is_cooperative) {
            // Define iteration parameters:
            unsigned int iter_lane_idx = 0;

            double iter_accum_etaSum{0};
            double iter_accum_phiSum{0};

            bool keep_accum = false;

            while (iter_lane_idx < eff_w_extent) {
              // Identify lanes whose remaining PFRecHitFraction work is exactly one element.
              // These lanes cannot benefit from recruiting cooperative helpers and are handled by a fast path.
              // Note: pfrhf_size may have been reduced in a previous iteration due to leftover handling.
              const warp::warp_mask_t single_worklane_mask = warp::ballot_mask(acc, active_lanes_mask, pfrhf_size == 1);

              // Create temporary per-iteration masks for cooperative scheduling:
              // -- how many vacant lanes in the warp, i.e., lanes available to become cooperators in this iteration ('free_lanes_mask')
              // -- how many reserved lanes in the warp ('swap_lanes_mask'); note that free_lanes_mask must also include reserved lanes
              warp::warp_mask_t free_lanes_mask = active_lanes_mask;
              warp::warp_mask_t swap_lanes_mask = static_cast<warp::warp_mask_t>(0);

              unsigned int swap_lane_idx{
                  w_extent};  //assigned to some default value (note that it's outside of the warp range)
              unsigned int swap_lanes_num{0};  //no lanes in the warp keep iter lane idx for swap operation

              unsigned int proc_lane_idx{w_extent};
              unsigned int proc_pfrhf_offset{static_cast<unsigned int>(pfrhf_offset)};

              bool update_params = true;

              bool load_flag = false;
              // Ensure masks and per-lane state updates from previous iteration are visible before scheduling.
              warp::syncWarpThreads_mask(acc, active_lanes_mask);

              while (update_params) {
                const warp::warp_mask_t iter_lane_mask = get_lane_mask(iter_lane_idx);

                const bool is_master_lane = iter_lane_idx == lane_idx;
                // first we need to check whether the current iter lane itself is vacant.
                if ((free_lanes_mask & iter_lane_mask) == 0) {
                  // The current iteration lane is not free: it currently holds state that must be preserved.
                  // Mark it as reserved-for-swap; later we move its state into an actually free lane.
                  swap_lanes_mask = swap_lanes_mask | iter_lane_mask;
                  ++swap_lanes_num;  // number of lanes reserved for swap in this iteration

                  if (is_master_lane && iter_lane_idx != proc_lane_idx)
                    swap_lane_idx = proc_lane_idx;
                } else {
                  // update the free mask (erase master-lane bit):
                  free_lanes_mask &= ~iter_lane_mask;
                }
                // 'iter_lane_idx' is warp-uniform; thus all lanes agree on which lane is the current master.
                // Check whether the current lane has exactly one element of work remaining.
                const bool is_single_work_lane = is_work_lane(single_worklane_mask, iter_lane_idx, w_extent);
                // Available cooperative subgroup capacity: free lanes minus those reserved to preserve state (swap lanes).
                // Note: the name 'subgroup' is used here because it does not take into account iterative (master) lane itself.
                const unsigned int free_subgroup_size = alpaka::popcount(acc, free_lanes_mask) - swap_lanes_num;

                if (is_single_work_lane) {
                  if (is_master_lane) {
                    proc_lane_idx = iter_lane_idx;
                    proc_pfrhf_offset = pfrhf_offset;
                    load_flag = true;
                  }
                  iter_lane_idx += 1;
                  update_params = iter_lane_idx < eff_w_extent &&
                                  (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
                  continue;
                }

                const unsigned int proc_pfrhf_size =
                    is_master_lane ? (pfrhf_size - 1) : 0;  //exclude master lane itself..
                // Broadcast worksize (all active lanes):
                const unsigned int iter_pfrhf_size =
                    warp::shfl_mask(acc, active_lanes_mask, proc_pfrhf_size, iter_lane_idx, w_extent);

                const unsigned int coop_subgroup_size = alpaka::math::min(acc, iter_pfrhf_size, free_subgroup_size);

                // Check which lane can cooperate in the work:
                // -- it must be vacant (corresponding bit in 'free_lanes_mask' must be set)
                // -- among free lanes, it must be within the first 'coop_subgroup_size' positions
                //    (using logical indexing over free_lanes_mask)
                // -- vacant lanes must not be reserved for swapping operation (which is already taking into account in 'coop_subgroup_size')
                const bool is_coop_subgroup_lane =
                    is_work_lane(free_lanes_mask, lane_idx, w_extent)
                        ? (get_logical_lane_idx(acc, free_lanes_mask, lane_idx) < coop_subgroup_size)
                        : false;
                // Cooperative subgroup lanes are drawn from free_lanes_mask; by construction this excludes the master lane
                // (because the master lane was already removed from free_lanes_mask earlier).
                const warp::warp_mask_t coop_subgroup_mask =
                    warp::ballot_mask(acc, active_lanes_mask, is_coop_subgroup_lane);
                // Erase corresponding bits in 'free_lanes_mask'
                free_lanes_mask &= ~coop_subgroup_mask;
                // Update parameters only for cooperative subgroup lanes (and the master lane):
                // Broadcast from the current master lane (iter_lane_idx): which lane owns the work and its current offset.
                // Only the cooperative subgroup lanes and the master lane participate in these shuffles
                if (is_coop_subgroup_lane || is_master_lane)
                  proc_lane_idx =
                      warp::shfl_mask(acc, coop_subgroup_mask | iter_lane_mask, iter_lane_idx, iter_lane_idx, w_extent);

                // Now we need to check whether we need to increment iteration lane index.
                // Check if worksize less or equal subgroup size, if 'true', increment iter lane index for the next rec hit fraction array:
                if (iter_pfrhf_size <= free_subgroup_size) {
                  iter_lane_idx += 1;
                  if (is_master_lane)
                    load_flag = true;  //master lane must store its values in the global arrays later
                } else if (is_master_lane) {
                  // Otherwise, the master lane has more work than available helpers; keep a leftover tail.
                  // We advance pfrhf_offset by (coop helpers + master) and reduce pfrhf_size accordingly.
                  keep_accum = true;
                  // update rechit fraction offset for the next iteration
                  proc_pfrhf_offset = pfrhf_offset;
                  pfrhf_offset += coop_subgroup_size + 1;  // +1 accounts for the master lane processing one element
                  // compute remaining work size (that is a leftover rec hit fraction size)
                  pfrhf_size = iter_pfrhf_size - coop_subgroup_size;
                }
                // Continue scheduling while we have remaining iter lanes and enough free capacity
                // to eventually place swap lanes
                update_params = (iter_lane_idx < eff_w_extent) &&
                                (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
              }

              // Now we need to swap cached values to vacant lanes:
              if (is_work_lane(free_lanes_mask | swap_lanes_mask, lane_idx, w_extent)) {
                const unsigned int src_log_lane_idx = is_work_lane(free_lanes_mask, lane_idx, w_extent)
                                                          ? get_logical_lane_idx(acc, free_lanes_mask, lane_idx)
                                                          : w_extent;
                const unsigned int src_phys_lane_idx =
                    src_log_lane_idx < swap_lanes_num ? get_physical_lane_idx(acc, swap_lanes_mask, src_log_lane_idx)
                                                      : lane_idx;

                const unsigned int tmp_proc_lane_idx =
                    warp::shfl_mask(acc, free_lanes_mask | swap_lanes_mask, swap_lane_idx, src_phys_lane_idx, w_extent);

                if (is_work_lane(free_lanes_mask, lane_idx, w_extent) && src_log_lane_idx < swap_lanes_num)
                  proc_lane_idx = tmp_proc_lane_idx;
              }

              const warp::warp_mask_t nonvacant_lanes_mask =
                  warp::ballot_mask(acc, active_lanes_mask, proc_lane_idx != w_extent);

              if (is_work_lane(nonvacant_lanes_mask, lane_idx, w_extent) == false)
                continue;

              const warp::warp_mask_t coop_group_mask = warp::match_any_mask(acc, nonvacant_lanes_mask, proc_lane_idx);

              const float proc_eta_c = warp::shfl_mask(acc, coop_group_mask, eta_c, proc_lane_idx, w_extent);
              const float proc_phi_c = warp::shfl_mask(acc, coop_group_mask, phi_c, proc_lane_idx, w_extent);
              const float proc_pfc_energy = warp::shfl_mask(acc, coop_group_mask, pfc_energy, proc_lane_idx, w_extent);

              const unsigned int coop_pfrhf_offset =
                  warp::shfl_mask(acc, coop_group_mask, proc_pfrhf_offset, proc_lane_idx, w_extent);

              const int pfrhfrac_idx = get_logical_lane_idx(acc, coop_group_mask, lane_idx) + coop_pfrhf_offset;

              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              const float frac_energy = pfRecHitFracs[pfrhfrac_idx].frac() * pfRecHit[pfrh_idx].energy();

              const double x_rh = pfRecHit[pfrh_idx].x();
              const double y_rh = pfRecHit[pfrh_idx].y();
              const double z_rh = pfRecHit[pfrh_idx].z();

              const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
              const float phi_rh = static_cast<float>(::cms::alpakatools::phi(acc, x_rh, y_rh));

              const double etaSum_ = static_cast<double>(frac_energy * alpaka::math::abs(acc, eta_rh - proc_eta_c));
              const double phiSum_ = static_cast<double>(
                  frac_energy * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, proc_phi_c)));

              iter_accum_etaSum += warp_sparse_reduce(acc, coop_group_mask, lane_idx, etaSum_, addFn);
              iter_accum_phiSum += warp_sparse_reduce(acc, coop_group_mask, lane_idx, phiSum_, addFn);

              if (load_flag) {
                const double etaRMS2_ = alpaka::math::max(acc, iter_accum_etaSum / proc_pfc_energy, rms2_threshold);
                mdpfClusteringVars[i].etaRMS2() = etaRMS2_ * etaRMS2_;

                const double phiRMS2_ = alpaka::math::max(acc, iter_accum_phiSum / proc_pfc_energy, rms2_threshold);
                mdpfClusteringVars[i].phiRMS2() = phiRMS2_ * phiRMS2_;
              }

              if (keep_accum == false) {
                iter_accum_etaSum = 0.;
                iter_accum_phiSum = 0.;
              }

              keep_accum = false;
            }  // end while
          } else {  //non cooperative work
            double accum_etaSum = 0.;
            double accum_phiSum = 0.;

            for (int pfrhfrac_idx = pfrhf_offset; pfrhfrac_idx < (pfrhf_offset + pfrhf_size); pfrhfrac_idx++) {
              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              const float frac = pfRecHitFracs[pfrhfrac_idx].frac();
              const float energy = pfRecHit[pfrh_idx].energy();

              const double x_rh = pfRecHit[pfrh_idx].x();
              const double y_rh = pfRecHit[pfrh_idx].y();
              const double z_rh = pfRecHit[pfrh_idx].z();

              const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
              const float phi_rh = static_cast<float>(::cms::alpakatools::phi(acc, x_rh, y_rh));

              auto etaSum_tmp = (frac * energy) * alpaka::math::abs(acc, eta_rh - eta_c);
              auto phiSum_tmp =
                  (frac * energy) * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, phi_c));

              accum_etaSum += etaSum_tmp;
              accum_phiSum += phiSum_tmp;
            }

            const double etaRMS2_ = alpaka::math::max(acc, accum_etaSum / pfc_energy, rms2_threshold);
            mdpfClusteringVars[i].etaRMS2() = etaRMS2_ * etaRMS2_;

            const double phiRMS2_ = alpaka::math::max(acc, accum_phiSum / pfc_energy, rms2_threshold);
            mdpfClusteringVars[i].phiRMS2() = phiRMS2_ * phiRMS2_;
          }
        }  //end uniform_groups
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
