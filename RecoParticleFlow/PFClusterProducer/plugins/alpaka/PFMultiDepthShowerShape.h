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
    using reduce_t = double;
    using compute_t = double;

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const reco::PFClusterDeviceCollection::ConstView pfClusters,
                                  const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHit,
                                  const float rms2_threshold = 0.1f) const {
      const unsigned int nClusters = pfClusters.nSeeds();

      if (::cms::alpakatools::once_per_grid(acc)) {
        mdpfClusteringVars.size() = nClusters;
      }

      const unsigned int w_extent = alpaka::warp::getSize(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks

        reduce_t accum_etaSum_div_en{0.};
        reduce_t accum_phiSum_div_en{0.};

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nClusters)) {
          const float pfc_energy = pfClusters[idx.global].energy();

          mdpfClusteringVars[idx.global].depth() = pfClusters[idx.global].depth();
          mdpfClusteringVars[idx.global].energy() = pfc_energy;

          const compute_t x_c = pfClusters[idx.global].x();
          const compute_t y_c = pfClusters[idx.global].y();
          const compute_t z_c = pfClusters[idx.global].z();

          const float eta_c = static_cast<float>(cms::alpakamath::eta(acc, x_c, y_c, z_c));
          const float phi_c = static_cast<float>(::cms::alpakatools::phi(acc, x_c, y_c));

          mdpfClusteringVars[idx.global].eta() = eta_c;
          mdpfClusteringVars[idx.global].phi() = phi_c;

          int pfrhf_offset = pfClusters[idx.global].rhfracOffset();
          int pfrhf_size = pfClusters[idx.global].rhfracSize();

          reduce_t iter_accum_etaSum{0};
          reduce_t iter_accum_phiSum{0};

          constexpr int pfrhf_size_threshold = 32;

          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const warp::warp_mask_t high_occup_lanes_mask =
              is_cooperative ? warp::ballot_mask(acc, active_lanes_mask, pfrhf_size > pfrhf_size_threshold) : 0;

          const unsigned int lane_idx = idx.local % w_extent;

          if (!is_cooperative || is_work_lane(high_occup_lanes_mask, lane_idx) == false) {
            for (int pfrhfrac_idx = pfrhf_offset; pfrhfrac_idx < (pfrhf_offset + pfrhf_size); pfrhfrac_idx++) {
              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              const float fracXenergy = pfRecHitFracs[pfrhfrac_idx].frac() * pfRecHit[pfrh_idx].energy();

              const compute_t x_rh = pfRecHit[pfrh_idx].x();
              const compute_t y_rh = pfRecHit[pfrh_idx].y();
              const compute_t z_rh = pfRecHit[pfrh_idx].z();

              const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
              const float phi_rh = static_cast<float>(::cms::alpakatools::phi(acc, x_rh, y_rh));

              auto etaSum_tmp = fracXenergy * alpaka::math::abs(acc, eta_rh - eta_c);
              auto phiSum_tmp = fracXenergy * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, phi_c));

              iter_accum_etaSum += etaSum_tmp;
              iter_accum_phiSum += phiSum_tmp;
            }

            accum_etaSum_div_en = iter_accum_etaSum / pfc_energy;
            accum_phiSum_div_en = iter_accum_phiSum / pfc_energy;
          }

          if constexpr (!is_cooperative)  //uniform condition for all lanes
            continue;

          const unsigned int eff_w_extent = alpaka::popcount(acc, high_occup_lanes_mask);

          auto addFn = [] ALPAKA_FN_ACC(reduce_t a, reduce_t b) -> reduce_t { return a + b; };

          // Define iteration parameters:
          unsigned int iter_lane_idx = 0;

          float iter_eta_c{0};
          float iter_phi_c{0};
          float iter_pfc_energy{0};

          bool update_iter_params = true;

          while (iter_lane_idx < eff_w_extent) {
            warp::syncWarpThreads_mask(acc, active_lanes_mask);

            const unsigned int src_lane_idx = get_physical_lane_idx(acc, high_occup_lanes_mask, iter_lane_idx);

            const unsigned int iter_pfrhf_size =
                warp::shfl_mask(acc, active_lanes_mask, pfrhf_size, src_lane_idx, w_extent);  // exclude the source lane

            const bool is_src_in_range = (src_lane_idx < iter_pfrhf_size);

            const bool is_coop_lane = lane_idx == src_lane_idx || (is_src_in_range && lane_idx < iter_pfrhf_size) ||
                                      (!is_src_in_range && lane_idx < (iter_pfrhf_size - 1));

            const unsigned int coop_group_mask = warp::ballot_mask(acc, active_lanes_mask, is_coop_lane);

            if (is_work_lane(coop_group_mask, lane_idx)) {
              if (update_iter_params) {
                iter_eta_c = warp::shfl_mask(acc, coop_group_mask, eta_c, src_lane_idx, w_extent);
                iter_phi_c = warp::shfl_mask(acc, coop_group_mask, phi_c, src_lane_idx, w_extent);
                iter_pfc_energy = warp::shfl_mask(acc, coop_group_mask, pfc_energy, src_lane_idx, w_extent);

                iter_accum_etaSum = 0;
                iter_accum_phiSum = 0;

                update_iter_params = false;
              }

              const unsigned int iter_pfrhf_offset_stub =
                  warp::shfl_mask(acc, coop_group_mask, pfrhf_offset, src_lane_idx, w_extent);

              const unsigned int stride =
                  (lane_idx != src_lane_idx || is_src_in_range) ? lane_idx : (iter_pfrhf_size - 1);

              const unsigned int pfrhfrac_idx = iter_pfrhf_offset_stub + stride;

              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              const float fracXenergy = pfRecHitFracs[pfrhfrac_idx].frac() * pfRecHit[pfrh_idx].energy();

              const compute_t x_rh = pfRecHit[pfrh_idx].x();
              const compute_t y_rh = pfRecHit[pfrh_idx].y();
              const compute_t z_rh = pfRecHit[pfrh_idx].z();

              const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
              const float phi_rh = static_cast<float>(::cms::alpakatools::phi(acc, x_rh, y_rh));

              const reduce_t etaSum_ = static_cast<reduce_t>(fracXenergy * alpaka::math::abs(acc, eta_rh - iter_eta_c));
              const reduce_t phiSum_ = static_cast<reduce_t>(
                  fracXenergy * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, iter_phi_c)));

              iter_accum_etaSum += warp_sparse_reduce(acc, coop_group_mask, lane_idx, etaSum_, addFn);
              iter_accum_phiSum += warp_sparse_reduce(acc, coop_group_mask, lane_idx, phiSum_, addFn);

              const unsigned int coop_group_size = alpaka::popcount(acc, coop_group_mask);

              if (iter_pfrhf_size == coop_group_size) {  //done for this iteration
                if (src_lane_idx == lane_idx) {
                  accum_etaSum_div_en = iter_accum_etaSum / iter_pfc_energy;
                  accum_phiSum_div_en = iter_accum_phiSum / iter_pfc_energy;
                }
                iter_lane_idx += 1;
                update_iter_params = true;
              } else {
                if (src_lane_idx == lane_idx) {
                  pfrhf_size -= coop_group_size;
                  pfrhf_offset += coop_group_size;
                }
              }
            } else {
              iter_lane_idx += 1;
              update_iter_params = true;
            }
          }  //end of while
        }  //end uniform_groups_elements

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nClusters)) {
          const reduce_t etaRMS2_ = alpaka::math::max(acc, accum_etaSum_div_en, rms2_threshold);
          const reduce_t phiRMS2_ = alpaka::math::max(acc, accum_phiSum_div_en, rms2_threshold);

          mdpfClusteringVars[idx.global].etaRMS2() = etaRMS2_ * etaRMS2_;
          mdpfClusteringVars[idx.global].phiRMS2() = phiRMS2_ * phiRMS2_;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
