#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthConstructLinks_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"

/**
 * @brief Warp-based link construction kernel for PF multi-depth clustering.
 *
 * Constructs a sparse directed link map from each destination (target) cluster to at most one
 * source cluster, using geometric proximity and energy criteria. The resulting link (stored as
 * mdpf_topoId per destination) is used as input to subsequent connected-components labeling
 * (i.e. ECL-CC algorithm).
 *
 * Execution model:
 * - Warp-tiling over (source tile × destination lane) pairs.
 * - Candidate filtering: depth > 0 and (eta, phi) within nSigma cuts.
 * - Multi-stage warp pruning using masked collectives (ballot/shuffle/reduction).
 *
 * Masking / portability notes:
 * - On SM80+ (Ampere and newer) warp collectives are issued with explicit masks that reflect
 *   the active candidate set plus the destination lane.
 * - On pre-SM80 CUDA targets we conservatively execute certain collectives with broader masks
 *   to avoid undefined behavior from masked participation constraints.
 *
 * Output:
 * - mdpfClusteringCCLabels[..].mdpf_topoId() is set to the selected source index, or to itself if
 *   no valid source is found (isolated node / self-link).
 */

// We perform full warp computation for old NVIDIA and AMD microarchitectures
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (__CUDA_ARCH__ < 800) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define FULL_WARP_COMPUTE
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::alpakatools;
  using namespace ::cms::alpakaintrinsics;

  // Link selection key used during multi-stage pruning
  enum class PFMDLinkParamKind { DZ, DR, ENERGY, INVALID_KIND };

  /**
 * @brief Lightweight register-cached view of per-cluster quantities used by the link builder.
 *
 * The default-constructed object represents an invalid/out-of-range ("ghost") cluster:
 * depth_ is set to lowest() so that depth tests fail deterministically.
 */

  class PFMDClusterParam {
  private:
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

      eta_ = cluster.eta();
      phi_ = cluster.phi();

      etaRMS2_ = cluster.etaRMS2();
      phiRMS2_ = cluster.phiRMS2();
    }

    constexpr float depth() const { return depth_; }
    constexpr float energy() const { return energy_; }
    constexpr float eta() const { return eta_; }
    constexpr float phi() const { return phi_; }

    constexpr double etaRMS2() const { return etaRMS2_; }
    constexpr double phiRMS2() const { return phiRMS2_; }
  };

  /**
 * @brief Link candidate (or selected link) with associated comparison keys.
 *
 * idx   : selected source cluster index (or self index as sentinel).
 * dz/dr : minimized keys; energy : maximized key used as final tie-breaker.
 */

  class PFMDLinkParam {
  private:
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

    constexpr float value(const PFMDLinkParamKind kind) const {
      if (kind == PFMDLinkParamKind::DZ) {
        return dz;
      } else if (kind == PFMDLinkParamKind::DR) {
        return dr;
      } else if (kind == PFMDLinkParamKind::ENERGY) {
        return energy;
      }
      return 0.f;
    }

    constexpr void try_update(const int new_idx, const float new_dz, const float new_dr, const float new_energy) {
      bool do_update = (dz > new_dz);

      if (dz == new_dz) {
        do_update = (dr > new_dr) ? true : (dr == new_dr ? (energy < new_energy) : false);
      }

      if (do_update) {
        idx = new_idx, dz = new_dz, dr = new_dr, energy = new_energy;
      }
    }

    constexpr int index() const { return idx; }
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

  /**
 * @brief Prune candidate links for a given destination lane using a single comparison key.
 *
 * The function reduces over the current candidate set and keeps only lanes that match the
 * winning value for the given key (min for DZ/DR, max for ENERGY). If the candidate set
 * collapses to a single lane, the destination lane updates 'dst_link_params' accordingly.
 *
 * @param acc              Alpaka accelerator instance.
 * @param mask             Active-lanes mask excluding the owner lane.
 * @param dst_link_params  Destination-selected link parameters (updated on the destination lane).
 * @param src_link_params  Per-lane candidate link parameters.
 * @param lane_idx         Index of the calling lane within the warp.
 * @param dst_lane_idx     Warp lane index that owns the destination cluster.
 * @param kind             Key used for pruning (DZ, DR, or ENERGY).
 * @param is_owner_tile    True when source==destination tile (used to early-exit if dst lane wins).
 *
 * @return New participation mask (still including destination lane). Returns 0 if selection has
 *         finished (winner chosen and dst updated, or destination lane is the winner in owner tile).
 */

  ALPAKA_FN_ACC ALPAKA_FN_INLINE warp::warp_mask_t prune_link(
      Acc1D const& acc,
      const warp::warp_mask_t mask,  //excludes the owner lane (corresponding bit set to 0)
      PFMDLinkParam& dst_link_params,
      const PFMDLinkParam& src_link_params,
      const unsigned int lane_idx,
      const unsigned int dst_lane_idx,
      const PFMDLinkParamKind kind,
      const bool is_owner_tile = false) {
    constexpr unsigned int w_extent = get_warp_size<Acc1D>();
    // 1. Create target lane mask:
    const warp::warp_mask_t dst_lane_mask = get_lane_mask(dst_lane_idx);
    // 2. First, we select parameter value for the selection process, based on specified test value type:
    const float val = src_link_params.value(kind);
    // 3. Do selection process (for active lanes specified in the mask):
    // 3.1 We need to find out the total number of active lanes.
    const unsigned int nLanes = alpaka::popcount(acc, mask);
    // 3.2. then check two cases: for a single active lane, just continue with the current mask, otherwise perform
    //      link filtering.
    warp::warp_mask_t leftover_mask = mask;
    //      Check number of active lanes and do filtering
    if (nLanes > 1) {
      // 3.3. Perform warp-level reduction:
      const float res_val =
          (kind == PFMDLinkParamKind::DZ || kind == PFMDLinkParamKind::DR)
              ? warp_sparse_reduce(acc, mask, lane_idx, val, CompFn<true>())
              : warp_sparse_reduce(acc, mask, lane_idx, val, CompFn<false>());  // for all lanes excl. owner dst lane!

      const unsigned int res_lane_idx = get_ls1b_idx(acc, mask);
      const float comp_val = warp::shfl_mask(acc, mask, res_val, res_lane_idx, w_extent);

      leftover_mask = warp::ballot_mask(acc, mask, (val == comp_val));
    }

    if (leftover_mask == dst_lane_mask && is_owner_tile)
      return 0;  // the destination lane is the winner, return zero mask
    // 4. If we have only one active lane:
    if (((leftover_mask & (leftover_mask - 1)) == 0)) {
      // 4.0 Compute the active lane index:
      const unsigned int res_lane_idx = get_ls1b_idx(acc, leftover_mask);

      const warp::warp_mask_t aggr_mask = leftover_mask | dst_lane_mask;
      // 4.1 Fetch new values from the source link:
      const float new_dz =
          warp::shfl_mask(acc, aggr_mask, src_link_params.value(PFMDLinkParamKind::DZ), res_lane_idx, w_extent);

      const float new_dr =
          warp::shfl_mask(acc, aggr_mask, src_link_params.value(PFMDLinkParamKind::DR), res_lane_idx, w_extent);

      const float new_energy =
          warp::shfl_mask(acc, aggr_mask, src_link_params.value(PFMDLinkParamKind::ENERGY), res_lane_idx, w_extent);

      const int new_idx = warp::shfl_mask(acc, aggr_mask, src_link_params.index(), res_lane_idx, w_extent);

      // 4.2. Try to update:
      if (lane_idx == dst_lane_idx) {
        dst_link_params.try_update(new_idx, new_dz, new_dr, new_energy);
      }
      return 0;
    }

    return leftover_mask;
  }
  /**
 * @brief Alpaka kernel constructing inter-cluster links for multi-depth clustering.
 *        Builds connectivity links between clusters based on precomputed clustering
 *        variables, producing connected-component labels for subsequent processing.
 *
 * @param acc                    Alpaka accelerator instance.
 * @param mdpfClusteringCCLabels  Output view for connected-component labels.
 * @param mdpfClusteringVars     Input view of multi-depth clustering variables.
 * @param nSigma                 Pointer to clustering parameter thresholds.
 */

  class ConstructLinksKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View mdpfClusteringCCLabels,
                                  const reco::PFMultiDepthClusteringVarsDeviceCollection::ConstView mdpfClusteringVars,
                                  const PFMultiDepthClusterParams* nSigma) const {
#ifdef FULL_WARP_COMPUTE
      constexpr bool full_warp_compute = true;
#else
      constexpr bool full_warp_compute = false;
#endif
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      const unsigned int nClusters = mdpfClusteringVars.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];

      const double nSigmaEta = nSigma->nSigmaEta;
      const double nSigmaPhi = nSigma->nSigmaPhi;
#ifndef FULL_WARP_COMPUTE
      constexpr PFMDLinkParamKind param_kinds[3] = {
          PFMDLinkParamKind::DZ, PFMDLinkParamKind::DR, PFMDLinkParamKind::ENERGY};
#endif
      if (::cms::alpakatools::once_per_grid(acc)) {
        mdpfClusteringCCLabels.size() = nClusters;
      }

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        constexpr unsigned int cluster_tile_size = w_extent;
        const unsigned int cluster_tiles =
            (std::is_same_v<Device, alpaka::DevCpu>) ? nClusters : (gridDim + (w_extent - 1)) / w_extent;
        // Execution domain along destination (target) clusters
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
          // Reset workl aux array:
          if (idx.global < nClusters) {
            mdpfClusteringCCLabels[idx.global].workl() = 0;
          }
/*
* For pre-Ampere CUDA architectures the operations must be done for all lanes in the whole warp.
*/
// Active-lane mask for this warp iteration:
// every warp collective must use a mask that covers all participating lanes;
// we additionally OR-in the destination lane to keep it active for result updates.
#ifndef FULL_WARP_COMPUTE
          const warp::warp_mask_t init_active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nClusters);
#endif
          // From this point all warp-level collectives must be accompanied with init_active_lanes_mask (or any derived from it) mask:
          // for example, new_mask = warp::ballot_mask(acc, old_mask, predicate) will generate a new mask that selects a subset of lanes from old_mask
          // Link parameters (by default store its own global index):
          PFMDLinkParam selected_link_params(idx.global);
          // Load destination (target) cluster parameters:
          const auto dst_cl_params =
              idx.global < nClusters ? PFMDClusterParam(mdpfClusteringVars[idx.global]) : PFMDClusterParam();
          // Get warp and lane indices
          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;
          // We iterate over source tiles (size = warp extent)
          // In fact, we process nCluster x nCluster domain, where we distribute tiles over the first dimention (row indices)
          // and warps over the second one (coloumn indices).
          // The first dimension corresponds to the "source" clusters, while the second to the destination
          // (or target) clusters. Note that the resulting link matrix is sparse with just a single entry per coloumn
          // (that means that each destination cluster may be linked to just a single source cluster)
          // but it may have up to nCluster-1 non-zero entries per row. Obviously, the matrix has zeros on the diagonal
          // i.e., self-linking is excluded.
          // For each destination lane (dst_lane_idx),
          // lanes in the current warp represent candidate sources within the current tile:
          //   src_idx = group*blockDim + tile*w_extent + lane_idx
          // The algorithm selects at most one source per destination and stores it into mdpf_topoId().

          for (unsigned int tile = 0; tile < cluster_tiles; tile++) {
            // We call a cluster tile as the 'owner' tile
            // if the lane index of each thread coincide with the destination cluster index
            // (in fact, we compare warp index with tile index):
            const bool is_owner_tile = (warp_idx + group * blockDim) == tile;
            // A destination cluster params, for 'non-owner' tile load cluster data again:
            const unsigned int src_idx = tile * cluster_tile_size + lane_idx;

            const auto src_cl_params =
                is_owner_tile
                    ? PFMDClusterParam(dst_cl_params)
                    : ((src_idx < nClusters) ? PFMDClusterParam(mdpfClusteringVars[src_idx]) : PFMDClusterParam());

            // Loop over lanes in the warp.
            // In fact, iteration lane index coincide with the target cluster index modulo warp extent (target cluster lane index)
            for (unsigned int dst_lane_idx = 0; dst_lane_idx < w_extent; dst_lane_idx++) {
              // 1. We need to keep the target cluster lane with dst_lane_idx reserved from divergence
              const warp::warp_mask_t dst_lane_mask = get_lane_mask(dst_lane_idx);
#ifndef FULL_WARP_COMPUTE
              const warp::warp_mask_t active_lanes_mask = init_active_lanes_mask | dst_lane_mask;
#endif
              const bool is_owner_lane = is_owner_tile && (dst_lane_idx == lane_idx);
              // 2. Broadcast values from dst_lane_idx, this will give us warp-local source cluster depth:
#ifndef FULL_WARP_COMPUTE
              const float dst_depth =
                  warp::shfl_mask(acc, active_lanes_mask, dst_cl_params.depth(), dst_lane_idx, w_extent);
#else
              const float dst_depth = alpaka::warp::shfl(acc, dst_cl_params.depth(), dst_lane_idx);
#endif
              // 3. Do not link at the same layer and only link inside out:
              //    Note that if lane_idx == iter_lane_id and is_proper_tile == true, then dz == 0 and the lane is filtered
              //   (but will be not excluded from active lanes)
              const auto dz = (static_cast<int>(dst_depth) - static_cast<int>(src_cl_params.depth()));
              // 4. Select lanes that contain valid candidates, i.e., all lanes for which dz > 0,
              //    excluding lane_idx = iter_lane_id and is_proper_tile = true
#ifndef FULL_WARP_COMPUTE
              warp::warp_mask_t leftover_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, dz > 0);
#else
              warp::warp_mask_t leftover_lanes_mask = alpaka::warp::ballot(acc, dz > 0);
#endif
              // 5. If the warp is 'empty' (no valid lanes), start the next iteration
              //    if no threads detected then coninue, no warp synchronization at the point
              //warp::syncWarpThreads_mask(acc, valid_candidates_mask || dst_lane_mask);
              // Note that lane with id equal to dst_lane_idx must be always active, since it's responsible for storing
              // result.
              // 6. Skip if there are no active lanes in the leftover lanes mask (all lanes are filtered and must skip);
              if (leftover_lanes_mask == 0)
                continue;
              // From here on, only lanes in (leftover_lanes_mask | dst_lane_mask) are allowed to call masked collectives.
#ifndef FULL_WARP_COMPUTE
              if (is_work_lane(leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent) == false)
                continue;

              const float dst_eta = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.eta(), dst_lane_idx, w_extent);
              const double dst_etaRMS2 = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.etaRMS2(), dst_lane_idx, w_extent);

              const auto tmp1 = src_cl_params.eta() - dst_eta;
              const auto deta = tmp1 * tmp1 / (src_cl_params.etaRMS2() + dst_etaRMS2);

              const float dst_phi = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.phi(), dst_lane_idx, w_extent);
              const double dst_phiRMS2 = warp::shfl_mask(
                  acc, leftover_lanes_mask | dst_lane_mask, dst_cl_params.phiRMS2(), dst_lane_idx, w_extent);

              const auto tmp2 = ::cms::alpakatools::deltaPhi(acc, src_cl_params.phi(), dst_phi);
              const auto dphi = tmp2 * tmp2 / (src_cl_params.phiRMS2() + dst_phiRMS2);
#else
              const float dst_eta = alpaka::warp::shfl(acc, dst_cl_params.eta(), dst_lane_idx);
              const double dst_etaRMS2 = alpaka::warp::shfl(acc, dst_cl_params.etaRMS2(), dst_lane_idx);

              const auto tmp1 = src_cl_params.eta() - dst_eta;
              const auto deta = is_work_lane(leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent)
                                    ? tmp1 * tmp1 / (src_cl_params.etaRMS2() + dst_etaRMS2)
                                    : 0;

              const float dst_phi = alpaka::warp::shfl(acc, dst_cl_params.phi(), dst_lane_idx);
              const double dst_phiRMS2 = alpaka::warp::shfl(acc, dst_cl_params.phiRMS2(), dst_lane_idx);

              const auto tmp2 = is_work_lane(leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent)
                                    ? ::cms::alpakatools::deltaPhi(acc, src_cl_params.phi(), dst_phi)
                                    : 0;
              const auto dphi = is_work_lane(leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent)
                                    ? tmp2 * tmp2 / (src_cl_params.phiRMS2() + dst_phiRMS2)
                                    : 0;
#endif
              const bool is_valid_lane =
                  (is_owner_lane == false) && is_work_lane(leftover_lanes_mask, lane_idx, w_extent);
#ifndef FULL_WARP_COMPUTE
              warp::warp_mask_t next_leftover_lanes_mask = warp::ballot_mask(
                  acc,
                  leftover_lanes_mask | dst_lane_mask,
                  (deta < nSigmaEta && dphi < nSigmaPhi) && is_valid_lane);  //update valid candidate mask
#else
              const bool pred = is_valid_lane && (deta < nSigmaEta && dphi < nSigmaPhi);
              warp::warp_mask_t next_leftover_lanes_mask =
                  alpaka::warp::ballot(acc, pred);  //update valid candidate mask
#endif

              if (next_leftover_lanes_mask == 0)
                continue;

              leftover_lanes_mask = next_leftover_lanes_mask | dst_lane_mask;
#ifndef FULL_WARP_COMPUTE
              if (is_work_lane(leftover_lanes_mask, lane_idx, w_extent) == false)
                continue;

              const float dst_energy =
                  warp::shfl_mask(acc, leftover_lanes_mask, dst_cl_params.energy(), dst_lane_idx, w_extent);
#else
              const float dst_energy = alpaka::warp::shfl(acc, dst_cl_params.energy(), dst_lane_idx);
#endif
              // 7. Now start inter-warp pruning:
              // 7.1 Create warp-local link params (with the latest leftover lane mask);
              const bool is_non_owner_lane = is_work_lane(next_leftover_lanes_mask, lane_idx, w_extent);

              auto candidate_link_params =
                  is_non_owner_lane
                      ? PFMDLinkParam(
                            src_idx, alpaka::math::abs(acc, dz), deta + dphi, dst_energy + src_cl_params.energy())
                      : PFMDLinkParam(idx.global);
              // 7.2 Check 3 parameters (dZ, dR, energy) to prune the candidate links:
              if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
                if (is_non_owner_lane)
                  selected_link_params.try_update(candidate_link_params.index(),
                                                  candidate_link_params.value(PFMDLinkParamKind::DZ),
                                                  candidate_link_params.value(PFMDLinkParamKind::DR),
                                                  candidate_link_params.value(PFMDLinkParamKind::ENERGY));
                continue;
              } else if constexpr (full_warp_compute) {
                for (unsigned int iter_idx = 0; iter_idx < w_extent; iter_idx++) {
                  const int flag = is_non_owner_lane ? 1 : 0;
                  const int is_valid_candidate = alpaka::warp::shfl(acc, flag, iter_idx);
                  if (is_valid_candidate) {
                    const int candidate_idx = alpaka::warp::shfl(acc, candidate_link_params.index(), iter_idx);
                    const float candidate_dz =
                        alpaka::warp::shfl(acc, candidate_link_params.value(PFMDLinkParamKind::DZ), iter_idx);
                    const float candidate_dr =
                        alpaka::warp::shfl(acc, candidate_link_params.value(PFMDLinkParamKind::DR), iter_idx);
                    const float candidate_energy =
                        alpaka::warp::shfl(acc, candidate_link_params.value(PFMDLinkParamKind::ENERGY), iter_idx);
                    if (lane_idx == dst_lane_idx)
                      selected_link_params.try_update(candidate_idx, candidate_dz, candidate_dr, candidate_energy);
                  }
                }
                continue;
              }
#ifndef FULL_WARP_COMPUTE
              for (unsigned int k = 0; k < 3; k++) {
                next_leftover_lanes_mask = prune_link(acc,
                                                      leftover_lanes_mask,
                                                      selected_link_params,
                                                      candidate_link_params,
                                                      lane_idx,
                                                      dst_lane_idx,
                                                      param_kinds[k],
                                                      is_owner_tile);
                if (next_leftover_lanes_mask == 0)
                  break;

                if (is_work_lane(next_leftover_lanes_mask | dst_lane_mask, lane_idx, w_extent) == false)
                  break;  // exit loop for filtered lanes only
                // Reset filtered link mask:
                leftover_lanes_mask = next_leftover_lanes_mask;
              }
#endif
            }  //end dst lane id
          }  //end all (full!) tiles
          // Store linked cluster id (or self index, if isolated)
          if (idx.global < nClusters) {
            mdpfClusteringCCLabels[idx.global].mdpf_topoId() = selected_link_params.index();
          }
        }  // end uniform_group_elements
      }  //end uniform_groups
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
