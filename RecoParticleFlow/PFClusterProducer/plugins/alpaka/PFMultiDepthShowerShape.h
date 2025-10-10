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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  using namespace alpaka_common;

  // epsilon^{1/4}:
  template <typename T>
    requires std::floating_point<T>
  constexpr T z_scaled() {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;

    if constexpr (std::is_same_v<U, float>) {
      constexpr float z_scaled_f = 53.81737057623773f;
      return z_scaled_f;
    } else {
      constexpr double z_scaled_d = 8192.0;
      return z_scaled_d;
    }
  }

  template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T GetPhi(TAcc const& acc, const T x, const T y) {
    return alpaka::math::atan2(acc, y, x);
  }

  template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T GetEta(TAcc const& acc, const T x, const T y, const T z) {
    // ROOT-style fast path:
    // uses log(zs + sqrt(zs^2 + 1)) when |zs| is moderate,
    // and a Taylor-aided form when |zs| is large.
    // For rho == 0, return +/-inf.

    T eta{0};

    const T rho = alpaka::math::sqrt(acc, x * x + y * y);

    if (rho > T{0}) {
      // threshold for switching to the Taylor-aided branch
      const T zscaled = z_scaled<T>();

      const T zs = z / rho;
      if (alpaka::math::abs(acc, zs) < zscaled) {
        eta = alpaka::math::log(acc, zs + alpaka::math::sqrt(acc, zs * zs + T{1}));
      } else {
        // first-order Taylor expansion for the sqrt part
        eta = (z > T{0}) ? alpaka::math::log(acc, T{2} * zs + T{0.5} / zs) : -alpaka::math::log(acc, -T{2} * zs);
      }
    } else {
      // Exactly along the beam axis:
      if (z != T{0})
        eta = alpaka::math::copysign(acc, std::numeric_limits<T>::infinity(), z);
    }
    return eta;
  }

  template <unsigned int max_w_items = 32>
  class ShowerShapeKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const reco::PFClusterDeviceCollection::ConstView pfClusters,
                                  const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHit) const {
      const unsigned int nClusters = pfClusters.size();

      const unsigned int w_extent = alpaka::warp::getSize(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int eff_w_extent = alpaka::popcount(acc, active_lanes_mask);
          // Skip inactive lanes:
          if (idx.global >= nClusters)
            continue;

          const int i = idx.global;

          const auto pfc_energy = pfClusters[i].energy();

          mdpfClusteringVars[i].depth() = pfClusters[i].depth();
          mdpfClusteringVars[i].energy() = pfc_energy;

          const auto x_c = pfClusters[i].x();
          const auto y_c = pfClusters[i].y();
          const auto z_c = pfClusters[i].z();

          const auto eta_c = GetEta(acc, x_c, y_c, z_c);
          const auto phi_c = GetPhi(acc, x_c, y_c);

          mdpfClusteringVars[i].eta() = eta_c;
          mdpfClusteringVars[i].phi() = phi_c;

          const int pfrhf_offset = pfClusters[i].rhfracOffset();
          const int pfrhf_size = pfClusters[i].rhfracSize();

          auto addFn = [] ALPAKA_FN_ACC(double a, double b) -> double { return a + b; };

          bool update_params = true;
          // Declare iteration parameters:
          unsigned int iter_lane_idx = 0;

          double iter_accum_etaSum, iter_accum_phiSum;

          unsigned int iter_pfrhf_offset, iter_pfrhf_size, iter_consumed_pfrhf_size;

          float iter_eta_c, iter_phi_c;

          while (iter_lane_idx < eff_w_extent) {
            warp::syncWarpThreads_mask(acc, active_lanes_mask);
            if (update_params) {
              iter_eta_c = warp::shfl_mask(acc, active_lanes_mask, eta_c, iter_lane_idx, w_extent);
              iter_phi_c = warp::shfl_mask(acc, active_lanes_mask, phi_c, iter_lane_idx, w_extent);

              iter_pfrhf_offset = warp::shfl_mask(acc, active_lanes_mask, pfrhf_offset, iter_lane_idx, w_extent);
              iter_pfrhf_size = warp::shfl_mask(acc, active_lanes_mask, pfrhf_size, iter_lane_idx, w_extent);

              iter_consumed_pfrhf_size = 0;

              iter_accum_etaSum = 0.;
              iter_accum_phiSum = 0.;

              update_params = false;
            }
            const int pfrhfrac_idx = iter_consumed_pfrhf_size + lane_idx + iter_pfrhf_offset;

            double etaSum_ = 0.;
            double phiSum_ = 0.;

            const unsigned int iter_leftover_pfrhf_size = iter_pfrhf_size - iter_consumed_pfrhf_size;  // check

            if (lane_idx < iter_leftover_pfrhf_size) {
              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();

              const float frac = pfRecHitFracs[pfrhfrac_idx].frac();

              const float energy = pfRecHit[pfrh_idx].energy();

              const auto x_rh = pfRecHit[pfrh_idx].x();
              const auto y_rh = pfRecHit[pfrh_idx].y();
              const auto z_rh = pfRecHit[pfrh_idx].z();

              const auto eta_rh = GetEta(acc, x_rh, y_rh, z_rh);
              const auto phi_rh = GetPhi(acc, x_rh, y_rh);

              etaSum_ = (frac * energy) * alpaka::math::abs(acc, eta_rh - iter_eta_c);
              phiSum_ = (frac * energy) * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, iter_phi_c));
            }

            warp::syncWarpThreads_mask(acc, active_lanes_mask);

            if (eff_w_extent == w_extent) {
              iter_accum_etaSum += warp_reduce(acc, etaSum_, addFn);
              iter_accum_phiSum += warp_reduce(acc, phiSum_, addFn);
            } else {
              iter_accum_etaSum += warp_sparse_reduce(acc, active_lanes_mask, lane_idx, etaSum_, addFn);
              iter_accum_phiSum += warp_sparse_reduce(acc, active_lanes_mask, lane_idx, phiSum_, addFn);
            }

            if (iter_leftover_pfrhf_size < eff_w_extent) {
              if (lane_idx == iter_lane_idx) {
                const double etaRMS2_ = alpaka::math::max(acc, iter_accum_etaSum / pfc_energy, 0.1);
                mdpfClusteringVars[i].etaRMS2() = etaRMS2_ * etaRMS2_;

                const double phiRMS2_ = alpaka::math::max(acc, iter_accum_phiSum / pfc_energy, 0.1);
                mdpfClusteringVars[i].phiRMS2() = phiRMS2_ * phiRMS2_;
              }

              iter_lane_idx += 1;

              update_params = true;
            } else {
              iter_consumed_pfrhf_size += eff_w_extent;
            }
          }
        }  //end uniform_groups
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
