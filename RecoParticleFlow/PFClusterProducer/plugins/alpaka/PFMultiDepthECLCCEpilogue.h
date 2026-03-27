#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h

#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

/**
 * @file PFMultiDepthECLCCEpilogue.h
 * @brief Warp-based postprocessing kernel for multi-depth particle flow clustering (ECL-CC Epilogue).
 *
 * This header defines and implements an Alpaka GPU kernel that finalizes the clustering of
 * particle flow clusters after connected components (ECL-CC) detection.
 *
 * The kernel performs:
 * - Consolidation of connected component membership for each cluster.
 * - Assignment of component energy sums based on rechit fractions.
 * - Remapping of cluster indices to component indices.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::alpakatools;
  using namespace ::cms::alpakaintrinsics;

  /**
 * @brief Alpaka kernel finalizing PF clusters after ECL-CC clustering.
 *
 * Produces the final PF cluster and rec hit fraction collections using the
 * connected-component labels computed by the ECL clustering stage. Supports
 * optional cooperative warp-level processing.
 *
 * @tparam max_w_items     Maximum number of items processed per warp.
 * @tparam is_cooperative Enable cooperative warp-level processing if true.
 * 
 * @param acc                 Alpaka accelerator object.
 * @param outPFCluster        Output PF cluster device collection.
 * @param outPFRecHitFracs    Output PF rec hit fraction device collection.
 * @param pfClusteringCCLabels Input view of connected-component labels.
 * @param pfCluster           Input PF cluster device collection.
 * @param pfRecHitFracs       Input PF rec hit fraction device collection.
 * @param pfRecHit            Input PF rec hit device collection.
 */

  template <unsigned int max_w_items = 32, bool is_cooperative = false>
  class ECLCCEpilogueKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFClusterDeviceCollection::View outPFCluster,
        reco::PFRecHitFractionDeviceCollection::View outPFRecHitFracs,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels,
        const reco::PFClusterDeviceCollection::ConstView pfCluster,
        const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
        const reco::PFRecHitDeviceCollection::ConstView pfRecHit) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      static_assert(max_w_items <= 32, "ECLCCEpilogueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      if constexpr (std::is_same_v<Device, alpaka::DevCpu> ||
                    std::is_same_v<alpaka::AccToTag<Acc1D>, alpaka::TagGpuHipRt>) {
        if (::cms::alpakatools::once_per_grid(acc)) {
          unsigned int ccrhfrac_idx = 0;
          unsigned int cc_idx = 0;

          for (unsigned int vtx_idx = 0; vtx_idx < nVertices; vtx_idx++) {
            const unsigned int rep_idx = pfClusteringCCLabels[vtx_idx].mdpf_topoId();
            const bool is_representative = rep_idx == vtx_idx;

            if (is_representative == false)
              continue;

            outPFCluster[cc_idx].depth() = pfCluster[rep_idx].depth();
            outPFCluster[cc_idx].topoId() = cc_idx;
            outPFCluster[cc_idx].energy() = pfCluster[rep_idx].energy();
            outPFCluster[cc_idx].x() = pfCluster[rep_idx].x();
            outPFCluster[cc_idx].y() = pfCluster[rep_idx].y();
            outPFCluster[cc_idx].z() = pfCluster[rep_idx].z();
            outPFCluster[cc_idx].topoRHCount() = pfCluster[rep_idx].topoRHCount();

            int cc_seed = pfCluster[rep_idx].seedRHIdx();
            float cc_energy = pfRecHit[cc_seed].energy();

            outPFCluster[cc_idx].rhfracOffset() = ccrhfrac_idx;

            for (unsigned int iter_idx = 0; iter_idx < nVertices; iter_idx++) {
              const unsigned int comp_id = pfClusteringCCLabels[iter_idx].mdpf_topoId();

              if (comp_id != rep_idx)
                continue;

              const int seed = pfCluster[iter_idx].seedRHIdx();
              const float energy = pfRecHit[seed].energy();

              const unsigned int rhf_begin = pfCluster[iter_idx].rhfracOffset();
              const unsigned int rhf_end = rhf_begin + pfCluster[iter_idx].rhfracSize();

              for (unsigned int src_rhfrac_idx = rhf_begin; src_rhfrac_idx < rhf_end; src_rhfrac_idx++) {
                const unsigned int dst_rhfrac_idx = ccrhfrac_idx;

                outPFRecHitFracs[dst_rhfrac_idx].frac() = pfRecHitFracs[src_rhfrac_idx].frac();
                outPFRecHitFracs[dst_rhfrac_idx].pfrhIdx() = pfRecHitFracs[src_rhfrac_idx].pfrhIdx();
                outPFRecHitFracs[dst_rhfrac_idx].pfcIdx() = cc_idx;

                ccrhfrac_idx += 1;
              }

              if (energy > cc_energy) {
                cc_energy = energy;
                cc_seed = seed;
              }
            }  // iter_idx

            outPFCluster[cc_idx].rhfracSize() = ccrhfrac_idx - outPFCluster[cc_idx].rhfracOffset();
            outPFCluster[cc_idx].seedRHIdx() = cc_seed;

            cc_idx += 1;  //nComponents
          }  // vtx_idx

          outPFCluster.nTopos() = cc_idx;
          outPFCluster.nSeeds() = cc_idx;
          outPFCluster.nRHFracs() = pfCluster.nRHFracs();
          outPFCluster.size() = cc_idx;
        }
        return;
      } else {
        // component offsets
        auto& subcc_offsets(alpaka::declareSharedVar<uint32_t[max_w_items], __COUNTER__>(acc));
        // isolate root flag : 0 - an isolated root vertex, 1 - a non-isolated root vertex
        auto& vertex_mask(alpaka::declareSharedVar<warp::warp_mask_t[max_w_items], __COUNTER__>(acc));

        auto& common_buf1(alpaka::declareSharedVar<uint32_t[max_w_items * w_extent], __COUNTER__>(acc));
        auto& common_buf2(alpaka::declareSharedVar<uint32_t[max_w_items * w_extent], __COUNTER__>(acc));
        auto& common_buf3(alpaka::declareSharedVar<uint32_t[max_w_items * w_extent], __COUNTER__>(acc));

        auto& cc_energy_seed(alpaka::declareSharedVar<uint64_t[max_w_items * w_extent], __COUNTER__>(acc));

        //block-local number of components
        unsigned int& localNComponents = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

        for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
          if (::cms::alpakatools::once_per_block(acc)) {
            localNComponents = 0;
          }

          unsigned int vertex_idx = nVertices;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            common_buf1[idx.local] = nVertices;
            common_buf2[idx.local] = 0;
            common_buf3[idx.local] = 0;

            cc_energy_seed[idx.local] = 0;

            if (idx.local < max_w_items) {
              subcc_offsets[idx.local] = 0;
              vertex_mask[idx.local] = active_lanes_mask;  //needed for AND operation
            }

            vertex_idx = idx.local;
          }

          const unsigned int rep_idx = vertex_idx < nVertices ? pfClusteringCCLabels[vertex_idx].mdpf_topoId() : 0;

          const bool is_representative = vertex_idx < nVertices ? vertex_idx == rep_idx : false;

          alpaka::syncBlockThreads(acc);

          //Maps vertex root -> compact component id (cc_idx); also uses [nVertices] as component counter:
          auto& component_map = common_buf1;
          auto& cc_idx_cache = common_buf3;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);
            const unsigned int lane_idx = idx.local % w_extent;

            const warp::warp_mask_t rep_mask = warp::ballot_mask(acc, active_lanes_mask, is_representative);

            if (is_representative == false)
              continue;

            const unsigned int low_rep_idx = get_ls1b_idx(acc, rep_mask);

            unsigned int local_topo_offset = 0;

            if (lane_idx == low_rep_idx) {
              const unsigned int local_n_topo = alpaka::popcount(acc, rep_mask);

              local_topo_offset = alpaka::atomicAdd(acc, &localNComponents, local_n_topo, alpaka::hierarchy::Threads{});
            }
            warp::syncWarpThreads_mask(acc, rep_mask);

            const unsigned int cc_idx_stub = warp::shfl_mask(acc, rep_mask, local_topo_offset, low_rep_idx, w_extent);
            const unsigned int cc_idx = cc_idx_stub + get_logical_lane_idx(acc, rep_mask, lane_idx);

            cc_idx_cache[cc_idx] = rep_idx;
            component_map[vertex_idx] = cc_idx;
          }
          alpaka::syncBlockThreads(acc);

          const unsigned int nComponents = localNComponents;
          const unsigned int rep_cc_idx = vertex_idx < nVertices ? component_map[rep_idx] : nVertices;

          // Store nComponents in the global buffer (per block):
          if (::cms::alpakatools::once_per_block(acc)) {
            outPFCluster.nTopos() = nComponents;
            outPFCluster.nSeeds() = nComponents;
            outPFCluster.nRHFracs() = pfCluster.nRHFracs();
            outPFCluster.size() = nComponents;
          }

          // Store relevant data (per thread):
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nComponents)) {
            const unsigned int topo_idx = idx.local;
            const unsigned int root_idx = cc_idx_cache[topo_idx];

            outPFCluster[topo_idx].depth() = pfCluster[root_idx].depth();
            outPFCluster[topo_idx].topoId() = topo_idx;
            outPFCluster[topo_idx].energy() = pfCluster[root_idx].energy();
            outPFCluster[topo_idx].x() = pfCluster[root_idx].x();
            outPFCluster[topo_idx].y() = pfCluster[root_idx].y();
            outPFCluster[topo_idx].z() = pfCluster[root_idx].z();
            outPFCluster[topo_idx].topoRHCount() = pfCluster[root_idx].topoRHCount();
            // reset buffer 3:
            common_buf3[topo_idx] = 0;
          }
          // total rhfrac count per component (accumulated at root).
          auto& connected_comp_rhf_sizes = common_buf2;

          unsigned int cc_rhf_relative_offset = 0;

          const unsigned int rhf_begin = vertex_idx < nVertices ? pfCluster[vertex_idx].rhfracOffset() : 0;
          const unsigned int rhf_size = vertex_idx < nVertices ? pfCluster[vertex_idx].rhfracSize() : 0;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int lane_idx = idx.local % w_extent;
            const warp::warp_mask_t rep_lane_mask = get_lane_mask(rep_idx % w_extent);

            const warp::warp_mask_t subcomp_mask = warp::match_any_mask(acc, active_lanes_mask, rep_idx);

            const unsigned int local_subcomponent_rep_idx = get_ls1b_idx(acc, subcomp_mask);

            bool update_iso_root_lane = false;

            unsigned int which_warp_idx = std::numeric_limits<unsigned int>::max();

            if (lane_idx == local_subcomponent_rep_idx) {
              const unsigned int subcomp_size = alpaka::popcount(acc, subcomp_mask);
              if ((subcomp_size > 1) || (subcomp_size == 1 && !is_representative)) {
                update_iso_root_lane = true;
                which_warp_idx = rep_idx / w_extent;
              }
            }

            const warp::warp_mask_t iso_root_lanes = warp::ballot_mask(acc, active_lanes_mask, update_iso_root_lane);

            if (iso_root_lanes != 0) {
// Avoid 'illigal instruction' crash on pre-Ampere GPUs
#if __CUDA_ARCH__ >= 800
              const warp::warp_mask_t iso_root_lanes_subgroup =
                  warp::match_any_mask(acc, iso_root_lanes, which_warp_idx);
#else
              // WARNING: this contains 'spurious' subgroup that must be filtered out
              const warp::warp_mask_t iso_root_lanes_subgroup =
                  warp::match_any_mask(acc, active_lanes_mask, which_warp_idx);
#endif
              // Construct correct rep mask:
              auto orFn = [] ALPAKA_FN_ACC(const warp::warp_mask_t m1,
                                           const warp::warp_mask_t m2) -> warp::warp_mask_t { return m1 | m2; };

              warp::warp_mask_t selected_iso_root_mask =
                  update_iso_root_lane ? warp_sparse_reduce(acc, iso_root_lanes_subgroup, lane_idx, rep_lane_mask, orFn)
                                       : 0;

              if (is_ls1b_idx<Acc1D>(iso_root_lanes_subgroup, lane_idx) && update_iso_root_lane) {
                // Temporary WAR (general form of De Morgan's law):
                const warp::warp_mask_t nonisolated_vertex_lanes = ~selected_iso_root_mask;

                alpaka::atomicAnd(
                    acc, &vertex_mask[which_warp_idx], nonisolated_vertex_lanes, alpaka::hierarchy::Threads{});
              }
            }

            unsigned int subcomp_rhf_offset = warp_sparse_exclusive_sum(acc, subcomp_mask, rhf_size, lane_idx);

            unsigned int relative_rhf_offset_stub = 0;

            if (lane_idx == local_subcomponent_rep_idx) {
              // Remark: exclusive sum returns total number of elements
              // (i.e., subcomponent rhf size) for the lowest lane idx in the mask.
              relative_rhf_offset_stub = alpaka::atomicAdd(
                  acc, &connected_comp_rhf_sizes[rep_idx], subcomp_rhf_offset, alpaka::hierarchy::Threads{});
              subcomp_rhf_offset = 0;  // we need to reset local offset for local rep lane.
            }

            const unsigned int relative_rhf_offset =
                warp::shfl_mask(acc, subcomp_mask, relative_rhf_offset_stub, local_subcomponent_rep_idx, w_extent);
            cc_rhf_relative_offset = relative_rhf_offset + subcomp_rhf_offset;  // store relative offsets
          }

          alpaka::syncBlockThreads(acc);

          // total rhfrac count per component (compact indexing).
          auto& cc_rhf_sizes = common_buf3;

          // Reset buffer 1:
          if (vertex_idx < max_w_items * w_extent)
            common_buf1[vertex_idx] = 0;

          if (is_representative) {
            cc_rhf_sizes[rep_cc_idx] = connected_comp_rhf_sizes[rep_idx];
          }

          alpaka::syncBlockThreads(acc);

          unsigned int cc_rhf_offset = 0;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nComponents)) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;

            const unsigned int cc_rhf_size = cc_rhf_sizes[idx.local];  //topo_id = idx.local
            // Store rhf sizes in memory:
            outPFCluster[idx.local].rhfracSize() = cc_rhf_size;

            // compute warp-local rhf offsets:
            cc_rhf_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, cc_rhf_size, lane_idx);

            if (lane_idx == 0) {
              subcc_offsets[warp_idx] = cc_rhf_offset;
              cc_rhf_offset = 0;
            }
          }
          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, w_extent)) {
            const unsigned int cc_rhf_size = subcc_offsets[idx.local];

            const unsigned int cc_rhf_global_offset = warp_exclusive_sum(acc, cc_rhf_size, idx.local);

            subcc_offsets[idx.local] = cc_rhf_global_offset;
          }

          alpaka::syncBlockThreads(acc);

          auto& cc_rhf_offsets = common_buf2;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nComponents)) {
            const unsigned int warp_idx = idx.local / w_extent;

            const unsigned int cc_rhf_global_offset = warp_idx == 0 ? 0 : subcc_offsets[warp_idx];

            if (warp_idx > 0)
              cc_rhf_offset += cc_rhf_global_offset;

            cc_rhf_offsets[idx.local] = cc_rhf_offset;
          }
          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nComponents)) {
            outPFCluster[idx.local].rhfracOffset() = cc_rhf_offset;
          }

          unsigned int vertex_seed = vertex_idx < nVertices ? pfCluster[vertex_idx].seedRHIdx() : 0;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int lane_idx = idx.local % w_extent;

            const unsigned int rhf_end = rhf_begin + rhf_size;

            unsigned int dst_rhf_offset = cc_rhf_relative_offset + cc_rhf_offsets[rep_cc_idx];

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
              const unsigned int eff_w_extent = alpaka::popcount(acc, active_lanes_mask);

              unsigned int src_rhf_offset = rhf_begin;
              unsigned int rhf_size = rhf_end - rhf_begin;
              // Define iteration parameters:
              unsigned int iter_lane_idx = 0;

              while (iter_lane_idx < eff_w_extent) {
                // Identify lanes whose remaining PFRecHitFraction work is exactly one element.
                // These lanes cannot benefit from recruiting cooperative helpers and are handled by a fast path.
                // Note: pfrhf_size may have been reduced in a previous iteration due to leftover handling.
                const warp::warp_mask_t single_worklane_mask = warp::ballot_mask(acc, active_lanes_mask, rhf_size == 1);

                // Create temporary per-iteration masks for cooperative scheduling:
                // -- how many vacant lanes in the warp, i.e., lanes available to become cooperators in this iteration ('free_lanes_mask')
                // -- how many reserved lanes in the warp ('swap_lanes_mask'); note that free_lanes_mask must also include reserved lanes
                warp::warp_mask_t free_lanes_mask = active_lanes_mask;
                warp::warp_mask_t swap_lanes_mask = static_cast<warp::warp_mask_t>(0);

                unsigned int swap_lane_idx{w_extent};  //assigned to some default value (it's outside of the warp range)
                unsigned int swap_lanes_num{0};        //no lanes in the warp keep iter lane idx for swap operation

                unsigned int proc_lane_idx{w_extent};
                unsigned int proc_dst_rhf_offset{dst_rhf_offset};
                unsigned int proc_src_rhf_offset{src_rhf_offset};

                bool update_params = true;

                warp::syncWarpThreads_mask(acc, active_lanes_mask);

                while (update_params) {
                  const warp::warp_mask_t iter_lane_mask = get_lane_mask(iter_lane_idx);

                  const bool is_master_lane = iter_lane_idx == lane_idx;
                  // first we need to check whether the current iter lane itself is vacant.
                  if ((free_lanes_mask & iter_lane_mask) == 0) {
                    // The current iteration lane is not free: it currently holds state that must be preserved.
                    // Mark it as reserved-for-swap; later we move its state into an actually free lane.
                    swap_lanes_mask = swap_lanes_mask | iter_lane_mask;
                    ++swap_lanes_num;  //how many lanes are reserved for swap

                    if (is_master_lane && iter_lane_idx != proc_lane_idx)
                      swap_lane_idx = proc_lane_idx;
                  } else {
                    // update the free mask (erase master-lane bit):
                    free_lanes_mask &= ~iter_lane_mask;
                  }
                  // 'iter_lane_idx' is warp-uniform; thus all lanes agree on which lane is the current master.
                  // Check whether the current lane has exactly one element of work remaining.
                  const bool is_single_work_lane = is_work_lane(single_worklane_mask, iter_lane_idx);
                  // Available cooperative subgroup capacity: free lanes minus those reserved to preserve state (swap lanes).
                  // Note: the name 'subgroup' is used here because it does not take into account iterative (master) lane itself.
                  const unsigned int free_subgroup_size =
                      static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) - swap_lanes_num;

                  if (is_single_work_lane) {
                    if (is_master_lane) {
                      proc_lane_idx = iter_lane_idx;
                      proc_dst_rhf_offset = dst_rhf_offset;
                      proc_src_rhf_offset = src_rhf_offset;
                    }
                    iter_lane_idx += 1;
                    update_params =
                        iter_lane_idx < eff_w_extent &&
                        (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
                    continue;
                  }

                  const unsigned int proc_rhf_size =
                      is_master_lane ? (rhf_size - 1) : 0;  //exclude master lane itself..
                  // Broadcast worksize (all active lanes):
                  const unsigned int iter_rhf_size =
                      warp::shfl_mask(acc, active_lanes_mask, proc_rhf_size, iter_lane_idx, w_extent);

                  const unsigned int coop_subgroup_size = alpaka::math::min(acc, iter_rhf_size, free_subgroup_size);

                  // Check which lane can cooperate in the work:
                  // -- it must be vacant (corresponding bit in 'free_lanes_mask' must be set)
                  // -- among free lanes, it must be within the first 'coop_subgroup_size' positions
                  //    (using logical indexing over free_lanes_mask)
                  // -- vacant lanes must not be reserved for swapping operation (which is already taking into account in 'coop_subgroup_size')
                  const bool is_coop_subgroup_lane =
                      is_work_lane(free_lanes_mask, lane_idx)
                          ? (get_logical_lane_idx(acc, free_lanes_mask, lane_idx) < coop_subgroup_size)
                          : false;
                  // Cooperative subgroup lanes are drawn from free_lanes_mask; by construction this excludes the master lane
                  // (because the master lane was already removed from free_lanes_mask earlier).
                  const warp::warp_mask_t coop_subgroup_mask = warp::ballot_mask(
                      acc, active_lanes_mask, is_coop_subgroup_lane);  //Note: it excludes master lane.
                  // Erase corresponding bits in 'free_lanes_mask'
                  free_lanes_mask &= ~coop_subgroup_mask;
                  // Update parameters only for cooperative subgroup lanes (and the master lane)
                  // if 'true' - do broadcast of iter. lane index and corresponding rechit offset from source (current iterative) lane:
                  if (is_coop_subgroup_lane || is_master_lane)
                    proc_lane_idx = warp::shfl_mask(
                        acc, coop_subgroup_mask | iter_lane_mask, iter_lane_idx, iter_lane_idx, w_extent);

                  // Now we need to check whether we need to increment iteration lane index.
                  // Check if worksize less or equal subgroup size, if 'true', increment iter lane index for the next rec hit fraction array:
                  if (iter_rhf_size <= free_subgroup_size) {
                    iter_lane_idx += 1;
                    // if 'false', we  have to update a leftover work for the current master lane (will continue in the next iteration):
                  } else if (is_master_lane) {
                    // update rechit fraction offset for the next iteration
                    proc_dst_rhf_offset = dst_rhf_offset;
                    dst_rhf_offset +=
                        coop_subgroup_size + 1;  //we need to take into account the master lane itself (hence "+1" here)
                    proc_src_rhf_offset = src_rhf_offset;
                    src_rhf_offset +=
                        coop_subgroup_size + 1;  //we need to take into account the master lane itself (hence "+1" here)
                    // compute remaining work size (that is a leftover rec hit fraction size)
                    rhf_size = iter_rhf_size - coop_subgroup_size;
                  }
                  update_params = (iter_lane_idx < eff_w_extent) &&
                                  (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
                }
                // Now we need to swap cached values to vacant lanes:
                if (is_work_lane(free_lanes_mask | swap_lanes_mask, lane_idx)) {
                  const unsigned int src_log_lane_idx = is_work_lane(free_lanes_mask, lane_idx)
                                                            ? get_logical_lane_idx(acc, free_lanes_mask, lane_idx)
                                                            : w_extent;
                  const unsigned int src_phys_lane_idx =
                      src_log_lane_idx < swap_lanes_num ? get_physical_lane_idx(acc, swap_lanes_mask, src_log_lane_idx)
                                                        : lane_idx;

                  const unsigned int tmp_proc_lane_idx = warp::shfl_mask(
                      acc, free_lanes_mask | swap_lanes_mask, swap_lane_idx, src_phys_lane_idx, w_extent);

                  if (is_work_lane(free_lanes_mask, lane_idx) && src_log_lane_idx < swap_lanes_num)
                    proc_lane_idx = tmp_proc_lane_idx;
                }

                const warp::warp_mask_t nonvacant_lanes_mask =
                    warp::ballot_mask(acc, active_lanes_mask, proc_lane_idx != w_extent);

                if (is_work_lane(nonvacant_lanes_mask, lane_idx) == false)
                  continue;

                const warp::warp_mask_t coop_group_mask =
                    warp::match_any_mask(acc, nonvacant_lanes_mask, proc_lane_idx);

                const float proc_cc_idx = warp::shfl_mask(acc, coop_group_mask, rep_cc_idx, proc_lane_idx, w_extent);

                const unsigned int coop_dst_rhf_offset =
                    warp::shfl_mask(acc, coop_group_mask, proc_dst_rhf_offset, proc_lane_idx, w_extent);
                const unsigned int coop_src_rhf_offset =
                    warp::shfl_mask(acc, coop_group_mask, proc_src_rhf_offset, proc_lane_idx, w_extent);

                const auto log_lane_idx = get_logical_lane_idx(acc, coop_group_mask, lane_idx);
                const int dst_rhfrac_idx = log_lane_idx + coop_dst_rhf_offset;
                const int src_rhfrac_idx = log_lane_idx + coop_src_rhf_offset;

                outPFRecHitFracs[dst_rhfrac_idx].frac() = pfRecHitFracs[src_rhfrac_idx].frac();
                outPFRecHitFracs[dst_rhfrac_idx].pfrhIdx() = pfRecHitFracs[src_rhfrac_idx].pfrhIdx();
                outPFRecHitFracs[dst_rhfrac_idx].pfcIdx() = proc_cc_idx;
              }
            } else {
              unsigned int dst_rhfrac_idx = dst_rhf_offset;
              for (unsigned int src_rhfrac_idx = rhf_begin; src_rhfrac_idx < rhf_end; src_rhfrac_idx++) {
                outPFRecHitFracs[dst_rhfrac_idx].frac() = pfRecHitFracs[src_rhfrac_idx].frac();
                outPFRecHitFracs[dst_rhfrac_idx].pfrhIdx() = pfRecHitFracs[src_rhfrac_idx].pfrhIdx();
                outPFRecHitFracs[dst_rhfrac_idx].pfcIdx() = rep_cc_idx;
                ++dst_rhfrac_idx;
              }
            }
          }

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;

            if (is_representative) {
              const warp::warp_mask_t lane_mask = get_lane_mask(lane_idx);
              const warp::warp_mask_t non_isolated_vertices = vertex_mask[warp_idx];
              const bool is_isolated_root = ((lane_mask & non_isolated_vertices));

              if (is_isolated_root) {
                cc_energy_seed[rep_cc_idx] = static_cast<uint64_t>(vertex_seed);
                continue;
              }
            }

            const warp::warp_mask_t updated_active_lanes_mask = alpaka::warp::activemask(acc);

            const float energy = pfRecHit[vertex_seed].energy();

            const warp::warp_mask_t subcomponent_mask = warp::match_any_mask(acc, updated_active_lanes_mask, rep_idx);

            auto compFn = [] ALPAKA_FN_ACC(const float a, const float b) -> float { return a > b ? a : b; };

            const float max_energy = warp_sparse_reduce(acc, subcomponent_mask, lane_idx, energy, compFn);
            // Equality on energy is intentional: max_energy is selected from lane values via shuffles/reduction,
            // so it is bitwise-equal to at least one participating lane's energy.
            const warp::warp_mask_t max_energy_lanes_mask =
                warp::ballot_mask(acc, subcomponent_mask, max_energy == energy);

            const unsigned int max_energy_lane_idx = alpaka::ffs(acc, static_cast<int>(max_energy_lanes_mask)) - 1;

            //need to store seed:
            const auto seed_max = warp::shfl_mask(acc, subcomponent_mask, vertex_seed, max_energy_lane_idx, w_extent);
            const auto energy_max = warp::shfl_mask(acc, subcomponent_mask, max_energy, max_energy_lane_idx, w_extent);

            // Energies are assumed non-negative; bit-cast uint ordering matches float ordering for atomicMax.
            if (is_ls1b_idx<Acc1D>(subcomponent_mask, lane_idx)) {
              uint32_t e_uint = std::bit_cast<unsigned int>(energy_max);
              uint64_t x = (static_cast<uint64_t>(e_uint) << 32) | static_cast<uint64_t>(seed_max);
              alpaka::atomicMax(acc, &cc_energy_seed[rep_cc_idx], x);
            }
          }

          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nComponents)) {
            const unsigned int topo_idx = idx.local;
            // Retrieve relevant seed :
            uint64_t energy_seed_packed = cc_energy_seed[topo_idx];
            uint32_t seed_to_store = static_cast<uint32_t>(energy_seed_packed & 0xFFFFFFFF);

            outPFCluster[topo_idx].seedRHIdx() = seed_to_store;
          }
        }
      }  //end of solo-block branch
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
