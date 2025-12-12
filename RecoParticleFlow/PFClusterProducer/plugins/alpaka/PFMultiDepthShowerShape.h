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

  namespace cms::alpakamath {

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
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T phi(TAcc const& acc, const T x, const T y) {
      return alpaka::math::atan2(acc, y, x);
    }

    template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T eta(TAcc const& acc, const T x, const T y, const T z) {
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
  template <unsigned int max_w_items = 32, bool is_cooperative = true>
  class ShowerShapeKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const reco::PFClusterDeviceCollection::ConstView pfClusters,
                                  const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHit,
                                  const float rms2_threshold = 0.1) const {
      //const unsigned int nClusters = pfClusters.size();
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
          const float phi_c = static_cast<float>(cms::alpakamath::phi(acc, x_c, y_c));

          mdpfClusteringVars[i].eta() = eta_c;
          mdpfClusteringVars[i].phi() = phi_c;

          const int pfrhf_offset = pfClusters[i].rhfracOffset();
          const int pfrhf_size = pfClusters[i].rhfracSize();

          auto addFn = [] ALPAKA_FN_ACC(double a, double b) -> double { return a + b; };

          if constexpr (is_cooperative) {
            bool update_params = true;
            // Declare iteration parameters:
            unsigned int iter_lane_idx = 0;

            double iter_accum_etaSum, iter_accum_phiSum;

            unsigned int iter_pfrhf_offset, iter_pfrhf_size, iter_consumed_pfrhf_size;

            float iter_eta_c, iter_phi_c;

            while (iter_lane_idx < eff_w_extent) {
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

                const double x_rh = pfRecHit[pfrh_idx].x();
                const double y_rh = pfRecHit[pfrh_idx].y();
                const double z_rh = pfRecHit[pfrh_idx].z();

                const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
                const float phi_rh = static_cast<float>(cms::alpakamath::phi(acc, x_rh, y_rh));

                etaSum_ = (frac * energy) * alpaka::math::abs(acc, eta_rh - iter_eta_c);
                phiSum_ = (frac * energy) * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, iter_phi_c));
              }

              if (eff_w_extent == w_extent) {  // NOTE that active mask is teken into account
                iter_accum_etaSum += warp_reduce(acc, etaSum_, addFn);
                iter_accum_phiSum += warp_reduce(acc, phiSum_, addFn);
              } else {
                iter_accum_etaSum += warp_sparse_reduce(acc, active_lanes_mask, lane_idx, etaSum_, addFn);
                iter_accum_phiSum += warp_sparse_reduce(acc, active_lanes_mask, lane_idx, phiSum_, addFn);
              }

              if (iter_leftover_pfrhf_size < eff_w_extent) {
                if (lane_idx == iter_lane_idx) {
                  const double etaRMS2_ = alpaka::math::max(acc, iter_accum_etaSum / pfc_energy, rms2_threshold);
                  mdpfClusteringVars[i].etaRMS2() = etaRMS2_ * etaRMS2_;

                  const double phiRMS2_ = alpaka::math::max(acc, iter_accum_phiSum / pfc_energy, rms2_threshold);
                  mdpfClusteringVars[i].phiRMS2() = phiRMS2_ * phiRMS2_;
                }

                iter_lane_idx += 1;

                update_params = true;
              } else {
                iter_consumed_pfrhf_size += eff_w_extent;
              }
            }  // end while
          } else { //non cooperative work
            double accum_etaSum = 0.;
            double accum_phiSum = 0.;
            for (int pfrhfrac_idx = pfrhf_offset; pfrhfrac_idx < pfrhf_size; pfrhfrac_idx++ ){
              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              const float frac = pfRecHitFracs[pfrhfrac_idx].frac();
              const float energy = pfRecHit[pfrh_idx].energy();

              const double x_rh = pfRecHit[pfrh_idx].x();
              const double y_rh = pfRecHit[pfrh_idx].y();
              const double z_rh = pfRecHit[pfrh_idx].z();

              const float eta_rh = static_cast<float>(cms::alpakamath::eta(acc, x_rh, y_rh, z_rh));
              const float phi_rh = static_cast<float>(cms::alpakamath::phi(acc, x_rh, y_rh));

              auto etaSum_tmp = (frac * energy) * alpaka::math::abs(acc, eta_rh - eta_c);
              auto phiSum_tmp = (frac * energy) * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, phi_c)); 
              
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
