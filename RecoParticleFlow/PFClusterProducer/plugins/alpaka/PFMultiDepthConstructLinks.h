#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"

/**
 * @brief Warp-based link construction kernel for Particle Flow (PF) multi-depth clustering.
 *
 * This header defines and implements an Alpaka kernel that constructs links between 
 * particle flow clusters based on geometric proximity and energy sharing criteria.
 * It prepares the cluster connectivity information for subsequent topological 
 * clustering (for connected components analysis via ECL-CC algorithm).
 * 
 * The kernel builds a sparse link map between destination and source clusters,
 * selection is performed by:
 *   - Minimizing depth difference;
 *   - Minimizing transverse distance;
 *   - Maximizing energy.
 *
 * All operations are performed at warp level with warp-masked operations
 * (ballot, shuffle, masked synchronization etc.).
 * 
 * Coputational steps:
 * - Warp tiling over source and destination cluster pairs.
 * - Candidate filtering based on dz > 0;
 * - Geometric filtering based on deta and dphi cuts;
 * - Multi-stage link selection (dz, dr, energy priority).
 * - Store final selected link into cluster's topology ID field.
 *
 * - Full dynamic warp masking is used: ballots and shuffles operate only on selected active lanes.
 * - This kernel does not rely on block-wide reductions or shared memory atomics.
 * - Designed for input cluster graphs with sparse connectivity :
 *   in particular, the destination cluster can be conneted only to at most one source cluster.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  using namespace alpaka_common;

  using namespace reco::pfClustering;

  enum class PFMDLinkParamKind { DZ, DR, ENERGY, INVALID_KIND };

  enum class PFMDClusterParamKind { DEPTH, ENERGY, ETA, PHI, ETA_RMS2, PHI_RMS2, INVALID_KIND };

  class PFMDClusterParam {
  protected:
    float depth_ = 0.f;
    float energy_ = 0.f;
    //
    float eta_ = 0.f;
    float phi_ = 0.f;
    //
    double etaRMS2_ = 0.;
    double phiRMS2_ = 0.;

  public:
    PFMDClusterParam() = default;
    //
    PFMDClusterParam(const PFMDClusterParam&) = default;

    template <typename TClusterVar>
    constexpr PFMDClusterParam(const TClusterVar& cluster) {
      // load cluster params :
      depth_ = cluster.depth();
      energy_ = cluster.energy();

      eta_ = cluster.eta();  //cluster.posrep()(1);
      phi_ = cluster.phi();  //cluster.posrep()(2);

      etaRMS2_ = cluster.etaRMS2();
      phiRMS2_ = cluster.phiRMS2();
    }

    template <PFMDClusterParamKind kind = PFMDClusterParamKind::INVALID_KIND>
    inline constexpr auto Get() const {
      static_assert(kind != PFMDClusterParamKind::INVALID_KIND,
                    "Invalid parameter kind passed to ClusterParam::Get method.\n");

      if constexpr (kind == PFMDClusterParamKind::DEPTH) {
        return depth_;
      } else if constexpr (kind == PFMDClusterParamKind::ENERGY) {
        return energy_;
      } else if constexpr (kind == PFMDClusterParamKind::ETA) {
        return eta_;
      } else if constexpr (kind == PFMDClusterParamKind::PHI) {
        return phi_;
      } else if constexpr (kind == PFMDClusterParamKind::ETA_RMS2) {
        return etaRMS2_;
      } else {
        return phiRMS2_;
      }
    }
  };

  class PFMDLinkParam {
  protected:
    int idx = -1;  // source cluster index,

    float dz = FLT_MAX;
    float dr = FLT_MAX;
    float energy = 0.f;

  public:
    //
    PFMDLinkParam() = default;
    //
    constexpr PFMDLinkParam(const int idx_) : idx(idx_) {};
    //
    constexpr PFMDLinkParam(const int idx_, const float dz_, const float dr_, const float energy_)
        : idx(idx_), dz(dz_), dr(dr_), energy(energy_) {}

    //
    inline constexpr float Get(const PFMDLinkParamKind kind) const {
      if (kind == PFMDLinkParamKind::DZ) {
        return dz;
      } else if (kind == PFMDLinkParamKind::DR) {
        return dr;
      } else if (kind == PFMDLinkParamKind::ENERGY) {
        return energy;
      }
      return 0.f;
    }

    inline constexpr int GetIdx() const { return idx; }
    //
    inline constexpr void Set(const int idx_, const float dz_, const float dr_, const float energy_) {
      this->idx = idx_;
      this->dz = dz_;
      this->dr = dr_;
      this->energy = energy_;
    }
  };

  template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC static unsigned int filter_links(TAcc const& acc,
                                                 const unsigned int mask,
                                                 const unsigned int dst_lane_mask,
                                                 const T val,  //this is a destination lane value (from the "owner" warp)
                                                 const PFMDLinkParamKind kind) {
    // 1. We need to find out the total number of active lanes.
    unsigned int nLanes = alpaka::popcount(acc, mask);
    // 2. Then check two cases:
    // For a single active lane, return (nop) :
    if (nLanes == 1) {
      return mask;
    }
    // For a multi-lane execution do warp-level reduction with a target reduction operation.
    const unsigned int w_extent = alpaka::warp::getSize(acc);
    // 3. Select compare type (min/max)
    const bool compute_min = (kind == PFMDLinkParamKind::DZ or kind == PFMDLinkParamKind::DR);
    //
    T src_val = val;
    // 4. Perform warp-level reduction
    CMS_UNROLL_LOOP
    for (int offset = w_extent / 2; offset > 0; offset /= 2) {
      const T neigh_val = warp::shfl_down_mask(acc, mask, src_val, offset, w_extent);  //
      //
      src_val = compute_min ? alpaka::math::min(acc, src_val, neigh_val) : alpaka::math::max(acc, src_val, neigh_val);
      //
      warp::syncWarpThreads_mask(acc, mask);
    }
    //
    warp::syncWarpThreads_mask(acc, mask | dst_lane_mask);
    // 6. Get test value:
    //    Although we could narrow down mask for just a few active lanes (or even a single one), we need to get
    //    the resulting mask for all initial lanes.
    const unsigned int src_lane_idx = get_ls1b_idx(acc, mask);
    const T test_val = warp::shfl_mask(acc, mask | dst_lane_mask, src_val, src_lane_idx, w_extent);
    // 7. Return updated link candidate mask: for the "owner" tile this should at least contain the destination lane bit,
    //                                        but in the most general case may contain bits for more than one candidate lanes
    const auto filtered_lanes_mask = warp::ballot_mask(acc, mask | dst_lane_mask, (val == test_val));
    //
    return filtered_lanes_mask;
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC static unsigned int prune_link(
      TAcc const& acc,
      const unsigned int mask,  //excludes the owner lane (corresponding bit set to 0)
      PFMDLinkParam& dst_link_params,
      const PFMDLinkParam src_link_params,
      const unsigned int lane_idx,
      const unsigned int dst_lane_idx,
      const PFMDLinkParamKind kind,
      const bool is_owner_tile = false) {
    // 1. Create target lane mask:
    const unsigned int dst_lane_mask = (1 << dst_lane_idx);
    // 2. First, we select parameter value for the selection process, based on specified checked value type:
    const float val = src_link_params.Get(kind);
    // 3. Do selection process (for active lanes specified in the mask):
    const unsigned int filtered_mask = filter_links(acc, mask, dst_lane_mask, val, kind);
    //
    warp::syncWarpThreads_mask(acc, mask | dst_lane_mask);
    //
    if (filtered_mask == dst_lane_mask and is_owner_tile)
      return 0;  // the destination lane is the winner, return zero mask
    //
    // 4. If we have only one active lane:
    if (((filtered_mask & (filtered_mask - 1)) == 0)) {
      const unsigned int w_extent = alpaka::warp::getSize(acc);
      // 4.0 Compute the active lane index:
      const unsigned int result_lane_idx = get_ls1b_idx(acc, filtered_mask);
      //
      const unsigned int aggregate_mask = filtered_mask | dst_lane_mask;
      //
      // 4.1 Fetch new values from source link:
      const float dz_ = src_link_params.Get(PFMDLinkParamKind::DZ);
      const float new_dz = warp::shfl_mask(acc, aggregate_mask, dz_, result_lane_idx, w_extent);
      //
      const float dr_ = src_link_params.Get(PFMDLinkParamKind::DR);
      const float new_dr = warp::shfl_mask(acc, aggregate_mask, dr_, result_lane_idx, w_extent);
      //
      const float en_ = src_link_params.Get(PFMDLinkParamKind::ENERGY);
      const float new_energy = warp::shfl_mask(acc, aggregate_mask, en_, result_lane_idx, w_extent);
      //
      const int idx_ = src_link_params.GetIdx();
      const int new_idx = warp::shfl_mask(acc, aggregate_mask, idx_, result_lane_idx, w_extent);
      //
      int flag = 0;
      //
      if (lane_idx == dst_lane_idx) {
        const float old_dz = dst_link_params.Get(PFMDLinkParamKind::DZ);
        //
        if (old_dz > new_dz) {
          flag = 1;
        } else if (old_dz == new_dz) {
          const float old_dr = dst_link_params.Get(PFMDLinkParamKind::DR);
          if (old_dr > new_dr) {
            flag = 1;
          } else if (old_dr == new_dr) {
            const float old_energy = dst_link_params.Get(PFMDLinkParamKind::ENERGY);
            if (old_energy < new_energy)
              flag = 1;
          }
        }
        if (flag == 1)
          dst_link_params.Set(new_idx, new_dz, new_dr, new_energy);
      }
      //
      warp::syncWarpThreads_mask(acc, mask | dst_lane_mask);

      return 0;
    }
    return filtered_mask;
  }

  template <unsigned int max_w_items = 32>
  class ConstructLinksKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const PFMultiDepthClusterParams* nSigma) const {
      const unsigned int nClusters = mdpfClusteringVars.size();
      //
      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      //
      const unsigned int w_extent = alpaka::warp::getSize(acc);
      const unsigned int w_items = alpaka::math::min(acc, nBlocks / w_extent, max_w_items);  //
      //
      const auto cluster_tiles = w_items;
      //
      // Helper lambda expression to identify active lane id:
      auto is_active_lane = [](const unsigned int mask, const unsigned int lid) -> bool { return ((mask >> lid) & 1); };
      //
      //
      const double nSigmaEta_ = nSigma->nSigmaEta;
      const double nSigmaPhi_ = nSigma->nSigmaPhi;

      constexpr PFMDLinkParamKind param_kinds[3] = {PFMDLinkParamKind::DZ, PFMDLinkParamKind::DR, PFMDLinkParamKind::ENERGY};

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        // Execution domain along destination (target) clusters
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);
          const unsigned int eff_w_extent = alpaka::popcount(acc, active_lanes_mask);
          // From this point all warp-level collectives must be accompanied with active_lanes_mask (or any derived from it) mask:
          // for example new_mask = warp::ballot_mask(acc, old_mask, predicate) will generate a new mask that selects a subset of lanes from old_mask
          if (idx.global >= nClusters)
            continue;
          //
          const auto cluster_tile_size = eff_w_extent;
          // Load destination (target) cluster parameters:
          const PFMDClusterParam dst_cluster_params(mdpfClusteringVars[idx.global]);
          // Link parameters (by default store its own global index):
          PFMDLinkParam selected_link_params(idx.global);
          // Get warp and lane indices
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;

          // Loop over all source cluster tiles.
          // In fact, we process nCluster x nCluster domain, where we distribute tiles over the first dimention and warps over the
          // second one. The first dimension corresponds to the "source" clusters, while the second to the destination
          // (or target) clusters. Note that the resulting link matrix is sparse with just a single entry per coloumn
          // (that means that each destination cluster may be linked to just a single source cluster)
          // but it may have up to nCluster-1 non-zero entries per row. Obviously, the matrix has zeros on the diagonal
          // (self-linking is excluded).
          for (unsigned int tile = 0; tile < cluster_tiles; tile++) {
            // We call a cluster tile as the 'owner' tile
            // if the lane index of each thread coincide with the destination cluster index
            // (but in fact we compare warp index with tile index):
            const bool is_owner_tile = (warp_idx + group * nBlocks) == (tile + group * cluster_tiles);
            // A destination cluster params, for 'non-owner' tile load cluster data again:
            const auto src_idx = (tile * cluster_tile_size + lane_idx) + group * nBlocks;
            //
            const auto src_cluster_params =
                is_owner_tile ? PFMDClusterParam(dst_cluster_params) : PFMDClusterParam(mdpfClusteringVars[src_idx]);
            //
            // Loop over lanes in the warp.
            // In fact, iteration lane index coincide with the target cluster index modulo warp extent (target cluster lane index)
            CMS_UNROLL_LOOP
            for (unsigned int iter_lane_idx = 0; iter_lane_idx < eff_w_extent; iter_lane_idx++) {
              // 1. Do warp sync for each iteration:
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              // .. but we need to keep the target cluster lane with iter_lane_idx reserved from divergence
              const unsigned dst_lane_mask = (1 << iter_lane_idx);
              const bool is_owner_lane = is_owner_tile and (iter_lane_idx == lane_idx);
              // 2. Broadcast values from iter_lane_idx, this will give us warp-local source cluster depth:
              const float depth = src_cluster_params.Get<PFMDClusterParamKind::DEPTH>();
              const float src_depth = warp::shfl_mask(acc, active_lanes_mask, depth, iter_lane_idx, w_extent);
              // 3. Do not link at the same layer and only link inside out:
              //    Note that if lane_idx == iter_lane_id and is_proper_tile == true, then dz == 0 and the lane is filtered
              //   (but will be not excluded from active lanes)
              const auto dz =
                  (static_cast<int>(src_depth) - static_cast<int>(dst_cluster_params.Get<PFMDClusterParamKind::DEPTH>()));
              // 4. Select lanes that contain valid candidates, i.e., all lanes for which dz > 0,
              //    excluding lane_idx = iter_lane_id and is_proper_tile = true
              unsigned int filtered_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, dz > 0);
              // 5. If the warp is 'empty' (no valid lanes), start the next iteration
              //    if no threads detected then coninue, no warp synchronization at the point
              //warp::syncWarpThreads_mask(acc, valid_candidates_mask || dst_lane_mask);
              // Note that lane with id equal to iter_lane_idx must be always active, since it's responsible for storing
              // result.
              // 6. Skip if :
              //            6.1)  there are no active lanes in the filtered mask (then all lanes must skip);
              if (filtered_lanes_mask == 0)
                continue;
              // here and in some places below we need to combine the destination lane
              // with the filtered mask to avoid undefined behavior:
              filtered_lanes_mask |= dst_lane_mask;
              //            6.2)  there are unwanted (inactive) lanes but keep the destination (target) lane active;
              if (is_active_lane(filtered_lanes_mask, lane_idx) == false)
                continue;

              warp::syncWarpThreads_mask(acc, filtered_lanes_mask);
              // WARNING: from this point only lanes selected in the filtered_lanes_mask plus destination lane are active in iteration.
              const float eta = src_cluster_params.Get<PFMDClusterParamKind::ETA>();
              const float src_eta = warp::shfl_mask(acc, filtered_lanes_mask, eta, iter_lane_idx, w_extent);

              const double eta_rms2 = src_cluster_params.Get<PFMDClusterParamKind::ETA_RMS2>();
              const double src_etaRMS2 = warp::shfl_mask(acc, filtered_lanes_mask, eta_rms2, iter_lane_idx, w_extent);
              //
              const auto tmp1 = dst_cluster_params.Get<PFMDClusterParamKind::ETA>() - src_eta;
              const auto deta = tmp1 * tmp1 / (dst_cluster_params.Get<PFMDClusterParamKind::ETA_RMS2>() + src_etaRMS2);
              //
              const float phi = src_cluster_params.Get<PFMDClusterParamKind::PHI>();
              const float src_phi = warp::shfl_mask(acc, filtered_lanes_mask, phi, iter_lane_idx, w_extent);

              const double phi_rms2 = src_cluster_params.Get<PFMDClusterParamKind::PHI_RMS2>();
              const double src_phiRMS2 = warp::shfl_mask(acc, filtered_lanes_mask, phi_rms2, iter_lane_idx, w_extent);
              //
              const auto tmp2 =
                  ::cms::alpakatools::deltaPhi(acc, dst_cluster_params.Get<PFMDClusterParamKind::PHI>(), src_phi);
              const auto dphi = tmp2 * tmp2 / (dst_cluster_params.Get<PFMDClusterParamKind::PHI_RMS2>() + src_phiRMS2);
              //
              warp::syncWarpThreads_mask(acc, filtered_lanes_mask);
              unsigned int next_filtered_lanes_mask =
                  warp::ballot_mask(acc,
                                    filtered_lanes_mask,
                                    (deta < nSigmaEta_ and dphi < nSigmaPhi_) and
                                        (is_owner_lane == false));  //update valid candidate mask
              //
              if (next_filtered_lanes_mask == 0x0)
                continue;

              filtered_lanes_mask = next_filtered_lanes_mask | dst_lane_mask;
              //
              if (is_active_lane(filtered_lanes_mask, lane_idx) == false)
                continue;

              warp::syncWarpThreads_mask(acc, filtered_lanes_mask);
              //
              const float energy = src_cluster_params.Get<PFMDClusterParamKind::ENERGY>();
              const float src_energy = warp::shfl_mask(acc, filtered_lanes_mask, energy, iter_lane_idx, w_extent);
              //
              // 7. Now start inter-warp pruning:
              // 7.1 Create warp-local link params (with the latest filtered lane mask);
              next_filtered_lanes_mask = is_owner_tile ? filtered_lanes_mask ^ dst_lane_mask : filtered_lanes_mask;

              PFMDLinkParam candidate_link_params;

              if (is_active_lane(next_filtered_lanes_mask, lane_idx)) {
                candidate_link_params = PFMDLinkParam(src_idx,
                                                  alpaka::math::abs(acc, dz),
                                                  deta + dphi,
                                                  src_energy + dst_cluster_params.Get<PFMDClusterParamKind::ENERGY>());
              } else {
                candidate_link_params = PFMDLinkParam(idx.global);
              }

              // 7.2 Check 3 parameters (dZ, dR, energy) to prune the candidate links:
              CMS_UNROLL_LOOP
              for (unsigned int k = 0; k < 3; k++) {
                // Reset filtered link mask:
                filtered_lanes_mask = next_filtered_lanes_mask;
                //
                warp::syncWarpThreads_mask(acc, filtered_lanes_mask | dst_lane_mask);
                //
                next_filtered_lanes_mask = prune_link(acc,
                                                      filtered_lanes_mask,
                                                      selected_link_params,
                                                      candidate_link_params,
                                                      lane_idx,
                                                      iter_lane_idx,
                                                      param_kinds[k],
                                                      is_owner_tile);
                //
                if (next_filtered_lanes_mask == 0)
                  continue;  //exit for all active lanes if the new mask is empty
                else if (is_active_lane(next_filtered_lanes_mask | dst_lane_mask, lane_idx) == false)
                  continue;  // exit loop for filtered lanes only
              }
              //
            }  //end dst lane id
          }  //end tile
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Store linked cluster id (or self index, if isolated)
          mdpfClusteringVars[idx.global].mdpf_topoId() = selected_link_params.GetIdx();
        }  // end uniform_group_elements
      }  //end uniform_groups
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
