#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

//#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
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

  enum class PFMDLinkParamKind { DZ, DR, ENERGY, INVALID_KIND };

  class PFMDClusterParam {
  protected:
    float depth_ = std::numeric_limits<float>::lowest();  //will disable all ghost (i.e. out-of-boundary) clusters
    float energy_ = 0.f;

    float eta_ = 0.f;
    float phi_ = 0.f;

    double etaRMS2_ = 0.;
    double phiRMS2_ = 0.;

  public:
    PFMDClusterParam() = default;
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

    constexpr float GetDepth() const { return depth_; }
    constexpr float GetEnergy() const { return energy_; }
    constexpr float GetEta() const { return eta_; }
    constexpr float GetPhi() const { return phi_; }

    constexpr double GetEtaRMS2() const { return etaRMS2_; }
    constexpr double GetPhiRMS2() const { return phiRMS2_; }
  };

  class PFMDLinkParam {
  protected:
    int idx = -1;  // source cluster index,

    float dz = std::numeric_limits<float>::max();
    float dr = std::numeric_limits<float>::max();
    float energy = 0.f;

  public:
    PFMDLinkParam() = default;

    // NOTE: by default, each cluster is self-connected
    constexpr PFMDLinkParam(const int idx_) : idx(idx_) {};

    constexpr PFMDLinkParam(const int idx_, const float dz_, const float dr_, const float energy_)
        : idx(idx_), dz(dz_), dr(dr_), energy(energy_) {}

    constexpr float Get(const PFMDLinkParamKind kind) const {
      if (kind == PFMDLinkParamKind::DZ) {
        return dz;
      } else if (kind == PFMDLinkParamKind::DR) {
        return dr;
      } else if (kind == PFMDLinkParamKind::ENERGY) {
        return energy;
      }
      return 0.f;
    }

    constexpr void TryUpdate(const int new_idx, const float new_dz, const float new_dr, const float new_energy) {
      bool do_update = (dz > new_dz);

      if (dz == new_dz) {
        do_update = (dr > new_dr) ? true : (dr == new_dr ? (energy < new_energy) : false);
      }

      if (do_update) {
        idx = new_idx, dz = new_dz, dr = new_dr, energy = new_energy;
      }
    }

    constexpr int GetIdx() const { return idx; }

    constexpr void Set(const int idx_, const float dz_, const float dr_, const float energy_) {
      this->idx = idx_;
      this->dz = dz_;
      this->dr = dr_;
      this->energy = energy_;
    }
  };

  //  Define operation type:
  template <bool comp_min>
  struct CompFn {
    ALPAKA_FN_ACC float operator()(float a, float b) const {
      if constexpr (comp_min)
        return a < b ? a : b;
      else
        return a > b ? a : b;
    }
  };

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static unsigned int prune_link(
      TAcc const& acc,
      const unsigned int mask,  //excludes the owner lane (corresponding bit set to 0)
      PFMDLinkParam& dst_link_params,
      const PFMDLinkParam& src_link_params,
      const unsigned int lane_idx,
      const unsigned int dst_lane_idx,
      const PFMDLinkParamKind kind,
      const bool is_owner_tile = false) {
    const unsigned int w_extent = alpaka::warp::getSize(acc);

    // 1. Create target lane mask:
    const unsigned int dst_lane_mask = (1 << dst_lane_idx);
    // 2. First, we select parameter value for the selection process, based on specified test value type:
    const float val = src_link_params.Get(kind);
    // 3. Do selection process (for active lanes specified in the mask):
    // 3.1 We need to find out the total number of active lanes.
    const unsigned int nLanes = alpaka::popcount(acc, mask);
    // 3.2. then check two cases: for a single active lane, just continue with the current mask, otherwise perform
    //      link filtering.
    unsigned int leftover_mask = mask;

    //      Check number of active lanes and do filtering
    if (nLanes > 1) {
      // 3.3. Perform warp-level reduction:
      const float res_val =
          (kind == PFMDLinkParamKind::DZ || kind == PFMDLinkParamKind::DR)
              ? warp_sparse_reduce(acc, mask, lane_idx, val, CompFn<true>())
              : warp_sparse_reduce(acc, mask, lane_idx, val, CompFn<false>());  // for all lanes excl. owner dst lane!

      warp::syncWarpThreads_mask(acc, mask);

      const unsigned int res_lane_idx = get_ls1b_idx(acc, mask);
      const float comp_val = warp::shfl_mask(acc, mask, res_val, res_lane_idx, w_extent);

      leftover_mask = warp::ballot_mask(acc, mask, (val == comp_val));
    }

    warp::syncWarpThreads_mask(acc, mask);

    if (leftover_mask == dst_lane_mask && is_owner_tile)
      return 0;  // the destination lane is the winner, return zero mask
    // 4. If we have only one active lane:
    if (((leftover_mask & (leftover_mask - 1)) == 0)) {
      // 4.0 Compute the active lane index:
      const unsigned int res_lane_idx = get_ls1b_idx(acc, leftover_mask);

      const unsigned int aggr_mask = leftover_mask | dst_lane_mask;
      // 4.1 Fetch new values from source link:
      const float new_dz =
          warp::shfl_mask(acc, aggr_mask, src_link_params.Get(PFMDLinkParamKind::DZ), res_lane_idx, w_extent);

      const float new_dr =
          warp::shfl_mask(acc, aggr_mask, src_link_params.Get(PFMDLinkParamKind::DR), res_lane_idx, w_extent);

      const float new_energy =
          warp::shfl_mask(acc, aggr_mask, src_link_params.Get(PFMDLinkParamKind::ENERGY), res_lane_idx, w_extent);

      const int new_idx = warp::shfl_mask(acc, aggr_mask, src_link_params.GetIdx(), res_lane_idx, w_extent);

      // 4.2. Try to update:
      if (lane_idx == dst_lane_idx) {
        dst_link_params.TryUpdate(new_idx, new_dz, new_dr, new_energy);
      }
      warp::syncWarpThreads_mask(acc, mask);
      return 0;
    }
    return (leftover_mask | dst_lane_mask);
  }

  template <unsigned int max_w_items = 32>
  class ConstructLinksKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
                                  const PFMultiDepthClusterParams* nSigma) const {
      const unsigned int nClusters = mdpfClusteringVars.size();

      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

      const unsigned int w_extent = alpaka::warp::getSize(acc);
      const unsigned int w_items = alpaka::math::min(acc, (blockDim + (w_extent - 1)) / w_extent, max_w_items);

      const double nSigmaEta_ = nSigma->nSigmaEta;
      const double nSigmaPhi_ = nSigma->nSigmaPhi;

      constexpr PFMDLinkParamKind param_kinds[3] = {
          PFMDLinkParamKind::DZ, PFMDLinkParamKind::DR, PFMDLinkParamKind::ENERGY};

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        const auto cluster_tiles = w_items;
        const auto cluster_tile_size = w_extent;
        // Execution domain along destination (target) clusters
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          const unsigned int init_active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);
          // From this point all warp-level collectives must be accompanied with init_active_lanes_mask (or any derived from it) mask:
          // for example new_mask = warp::ballot_mask(acc, old_mask, predicate) will generate a new mask that selects a subset of lanes from old_mask
          // Link parameters (by default store its own global index):
          PFMDLinkParam selected_link_params(idx.global);
          // Load destination (target) cluster parameters:
          const auto dst_cl_params =
              idx.global < nClusters ? PFMDClusterParam(mdpfClusteringVars[idx.global]) : PFMDClusterParam();
          // Get warp and lane indices
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          // Loop over all source cluster tiles.
          // In fact, we process nCluster x nCluster domain, where we distribute tiles over the first dimention (row indices)
          // and warps over the second one (coloumn indices).
          // The first dimension corresponds to the "source" clusters, while the second to the destination
          // (or target) clusters. Note that the resulting link matrix is sparse with just a single entry per coloumn
          // (that means that each destination cluster may be linked to just a single source cluster)
          // but it may have up to nCluster-1 non-zero entries per row. Obviously, the matrix has zeros on the diagonal
          // i.e., self-linking is excluded.
          for (unsigned int tile = 0; tile < cluster_tiles; tile++) {
            // We call a cluster tile as the 'owner' tile
            // if the lane index of each thread coincide with the destination cluster index
            // (in fact, we compare warp index with tile index):
            const bool is_owner_tile = (warp_idx + group * blockDim) == (tile + group * cluster_tiles);
            // A destination cluster params, for 'non-owner' tile load cluster data again:
            const auto src_idx = (tile * cluster_tile_size + lane_idx) + group * blockDim;

            const auto src_cl_params =
                is_owner_tile
                    ? PFMDClusterParam(dst_cl_params)
                    : ((src_idx < nClusters) ? PFMDClusterParam(mdpfClusteringVars[src_idx]) : PFMDClusterParam());
            // Loop over lanes in the warp.
            // In fact, iteration lane index coincide with the target cluster index modulo warp extent (target cluster lane index)
            for (unsigned int dst_lane_idx = 0; dst_lane_idx < w_extent; dst_lane_idx++) {
              // 0. We need to keep the target cluster lane with dst_lane_idx reserved from divergence
              const unsigned dst_lane_mask = (1 << dst_lane_idx);
              const unsigned int active_lanes_mask = init_active_lanes_mask | dst_lane_mask;
              // 1. Do warp sync for each iteration:
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              const bool is_owner_lane = is_owner_tile && (dst_lane_idx == lane_idx);
              // 2. Broadcast values from dst_lane_idx, this will give us warp-local source cluster depth:
              const float dst_depth =
                  warp::shfl_mask(acc, active_lanes_mask, dst_cl_params.GetDepth(), dst_lane_idx, w_extent);
              // 3. Do not link at the same layer and only link inside out:
              //    Note that if lane_idx == iter_lane_id and is_proper_tile == true, then dz == 0 and the lane is filtered
              //   (but will be not excluded from active lanes)
              const auto dz = (static_cast<int>(dst_depth) - static_cast<int>(src_cl_params.GetDepth()));
              // 4. Select lanes that contain valid candidates, i.e., all lanes for which dz > 0,
              //    excluding lane_idx = iter_lane_id and is_proper_tile = true
              unsigned int leftover_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, dz > 0);
              // 5. If the warp is 'empty' (no valid lanes), start the next iteration
              //    if no threads detected then coninue, no warp synchronization at the point
              //warp::syncWarpThreads_mask(acc, valid_candidates_mask || dst_lane_mask);
              // Note that lane with id equal to dst_lane_idx must be always active, since it's responsible for storing
              // result.
              // 6. Skip if :
              //            6.1)  there are no active lanes in the leftover lanes mask (all lanes are filtered and must skip);
              if (leftover_lanes_mask == 0)
                continue;
              // here and in some places below we need to combine the destination lane
              // with the leftover mask to avoid undefined behavior:
              if (is_work_lane(leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent) == false)
                continue;

              warp::syncWarpThreads_mask(acc, leftover_lanes_mask | dst_lane_mask);
              // WARNING: from this point only lanes selected in the leftover_lanes_mask plus destination lane are active in iteration.
              const float dst_eta = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.GetEta(), dst_lane_idx, w_extent);
              const double dst_etaRMS2 = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.GetEtaRMS2(), dst_lane_idx, w_extent);

              const auto tmp1 = src_cl_params.GetEta() - dst_eta;
              const auto deta = tmp1 * tmp1 / (src_cl_params.GetEtaRMS2() + dst_etaRMS2);

              const float dst_phi = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.GetPhi(), dst_lane_idx, w_extent);
              const double dst_phiRMS2 = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.GetPhiRMS2(), dst_lane_idx, w_extent);

              const auto tmp2 = ::cms::alpakatools::deltaPhi(acc, src_cl_params.GetPhi(), dst_phi);
              const auto dphi = tmp2 * tmp2 / (src_cl_params.GetPhiRMS2() + dst_phiRMS2);
              warp::syncWarpThreads_mask(acc, leftover_lanes_mask | dst_lane_mask);

              const bool is_valid_lane =
                  (is_owner_lane == false) && is_work_lane(leftover_lanes_mask, lane_idx, w_extent);

              unsigned int next_leftover_lanes_mask = warp::ballot_mask(
                  acc,
                  leftover_lanes_mask | dst_lane_mask,
                  (deta < nSigmaEta_ && dphi < nSigmaPhi_) && is_valid_lane);  //update valid candidate mask

              if (next_leftover_lanes_mask == 0)
                continue;

              leftover_lanes_mask = next_leftover_lanes_mask | dst_lane_mask;

              if (is_work_lane(leftover_lanes_mask, lane_idx, w_extent) == false)
                continue;

              warp::syncWarpThreads_mask(acc, leftover_lanes_mask);

              const float dst_energy =
                  warp::shfl_mask(acc, leftover_lanes_mask, dst_cl_params.GetEnergy(), dst_lane_idx, w_extent);
              // 7. Now start inter-warp pruning:
              // 7.1 Create warp-local link params (with the latest leftover lane mask);
              const bool is_non_owner_lane = is_work_lane(next_leftover_lanes_mask, lane_idx, w_extent);

              auto candidate_link_params =
                  is_non_owner_lane
                      ? PFMDLinkParam(
                            src_idx, alpaka::math::abs(acc, dz), deta + dphi, dst_energy + src_cl_params.GetEnergy())
                      : PFMDLinkParam(idx.global);
              // 7.2 Check 3 parameters (dZ, dR, energy) to prune the candidate links:
              for (unsigned int k = 0; k < 3; k++) {
                warp::syncWarpThreads_mask(acc, leftover_lanes_mask);

                next_leftover_lanes_mask = prune_link(acc,
                                                      leftover_lanes_mask,
                                                      selected_link_params,
                                                      candidate_link_params,
                                                      lane_idx,
                                                      dst_lane_idx,
                                                      param_kinds[k],
                                                      is_owner_tile);

                if (is_work_lane(next_leftover_lanes_mask, lane_idx, w_extent) == false)
                  break;  // exit loop for filtered lanes only
                // Reset filtered link mask:
                leftover_lanes_mask = next_leftover_lanes_mask;
              }
            }  //end dst lane id
          }  //end all (full!) tiles
          warp::syncWarpThreads_mask(acc, init_active_lanes_mask);
          // Store linked cluster id (or self index, if isolated)

          if (idx.global < nClusters)
            mdpfClusteringVars[idx.global].mdpf_topoId() = selected_link_params.GetIdx();

        }  // end uniform_group_elements
      }  //end uniform_groups
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
