#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthShowerShape_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthShowerShape_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterizerHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  using namespace alpaka_common;

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
      //
      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      //
      const unsigned int w_extent = alpaka::warp::getSize(acc);
      //
      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);
          //
          const unsigned int lane_idx = idx.local % w_extent;
          //
          const unsigned int warp_work_extent = alpaka::popcount(acc, active_lanes_mask);
          // Skip inactive lanes:
          if (idx.global >= nClusters)
            continue;
          //
          const int i = idx.global;
          //
          const auto pfc_energy = pfClusters[i].energy();
          //
          mdpfClusteringVars[i].depth() = pfClusters[i].depth();
          mdpfClusteringVars[i].energy() = pfc_energy;

          const auto eta_c = pfClusters[i].y();
          const auto phi_c = pfClusters[i].z();

          mdpfClusteringVars[i].eta() = eta_c;
          mdpfClusteringVars[i].phi() = phi_c;

          const int pfrhf_offset = pfClusters[i].rhfracOffset();
          const int pfrhf_size = pfClusters[i].rhfracSize();
          //
          auto addFn = [] ALPAKA_FN_ACC(double a, double b) -> double { return a + b; };
          //
          bool update_params = true;
          // Declare iteration parameters:
          unsigned int iter_lane_idx = 0;
          //
          double iter_accum_etaSum, iter_accum_phiSum;
          //
          unsigned int iter_pfrhf_offset, iter_pfrhf_size, iter_leftover_pfrhf_size;
          //
          while (iter_lane_idx < warp_work_extent) {
            if (update_params) {
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              //
              iter_pfrhf_offset = warp::shfl_mask(acc, active_lanes_mask, pfrhf_offset, iter_lane_idx, w_extent);
              iter_pfrhf_size = warp::shfl_mask(acc, active_lanes_mask, pfrhf_size, iter_lane_idx, w_extent);
              iter_leftover_pfrhf_size = 0;
              //
              iter_accum_etaSum = 0.;
              iter_accum_phiSum = 0.;
              //
              update_params = false;
            }
            const int pfrhfrac_idx = lane_idx + iter_pfrhf_offset + iter_leftover_pfrhf_size;
            //
            double etaSum_ = 0.;
            double phiSum_ = 0.;
            //
            if (lane_idx < (iter_pfrhf_size - iter_leftover_pfrhf_size)) {
              const int pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              //
              const float frac = pfRecHitFracs[pfrhfrac_idx].frac();
              //
              const float energy = pfRecHit[pfrh_idx].energy();
              //
              const auto eta_rh = pfRecHit[pfrh_idx].y();
              const auto phi_rh = pfRecHit[pfrh_idx].z();
              //
              etaSum_ = (frac * energy) * alpaka::math::abs(acc, eta_rh - eta_c);
              phiSum_ = (frac * energy) * alpaka::math::abs(acc, ::cms::alpakatools::deltaPhi(acc, phi_rh, phi_c));
            }
            //
            iter_accum_etaSum += warp_reduce<TAcc, double>(acc, active_lanes_mask, etaSum_, addFn);
            iter_accum_phiSum += warp_reduce<TAcc, double>(acc, active_lanes_mask, phiSum_, addFn);
            //
            if ((iter_pfrhf_size - iter_leftover_pfrhf_size) < warp_work_extent) {
              //
              if (lane_idx == iter_lane_idx) {
                const double etaRMS2_ = alpaka::math::max(acc, iter_accum_etaSum / pfc_energy, 0.1);
                mdpfClusteringVars[i].etaRMS2() = etaRMS2_ * etaRMS2_;
                //
                const double phiRMS2_ = alpaka::math::max(acc, iter_accum_phiSum / pfc_energy, 0.1);
                mdpfClusteringVars[i].phiRMS2() = phiRMS2_ * phiRMS2_;
              }
              //
              iter_lane_idx += 1;
              //
              update_params = true;
            } else {
              iter_leftover_pfrhf_size += warp_work_extent;
            }
          }
        }  //end uniform_groups
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
