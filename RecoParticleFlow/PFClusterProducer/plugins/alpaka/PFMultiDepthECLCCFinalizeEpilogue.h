#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCFinalizeEpilogue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCFinalizeEpilogue_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCEpilogueArgsDeviceCollection.h"

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
  class ECLCCFinalizeEpilogueKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFClusterDeviceCollection::View outPFCluster,
        reco::PFRecHitFractionDeviceCollection::View outPFRecHitFracs,
        reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels,
        const reco::PFClusterDeviceCollection::ConstView pfCluster,
        const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
        const reco::PFRecHitDeviceCollection::ConstView pfRecHit) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      static_assert(max_w_items <= 32, "ECLCCEpilogueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

      auto& cc_energy_seed(alpaka::declareSharedVar<uint64_t[max_w_items * w_extent], __COUNTER__>(acc));

      //block-local number of components
      unsigned int& nComponents = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      unsigned int& blockRHFShift = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        if (::cms::alpakatools::once_per_block(acc)) {
          nComponents = outPFCluster.nSeeds();
          blockRHFShift = args[group * blockDim].blockRHFOffset();
        }

        unsigned int vertex_idx = nVertices;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          cc_energy_seed[idx.local] = 0;
          vertex_idx = idx.global;
        }
        alpaka::syncBlockThreads(acc);

        const unsigned int rep_idx = vertex_idx < nVertices ? pfClusteringCCLabels[vertex_idx].mdpf_topoId() : 0;

        const bool is_representative = vertex_idx < nVertices ? vertex_idx == rep_idx : false;

        bool is_isolated_root = false;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const unsigned int lane_idx = idx.local % w_extent;
          const warp::warp_mask_t non_isolated_vertices = is_representative ? args[rep_idx / w_extent].vertexMask() : 0;

          if (is_representative) {
            const warp::warp_mask_t lane_mask = get_lane_mask(lane_idx);
            is_isolated_root = ((lane_mask & non_isolated_vertices));
          }
        }

        const unsigned int global_topo_idx = vertex_idx < nVertices ? args[rep_idx].rootMap() : nVertices;
        const unsigned int rep_cc_idx = vertex_idx < nVertices ? args[rep_idx].rootLocalMap() : nVertices;

        const unsigned int rhf_begin = vertex_idx < nVertices ? pfCluster[vertex_idx].rhfracOffset() : 0;
        const unsigned int rhf_size = vertex_idx < nVertices ? pfCluster[vertex_idx].rhfracSize() : 0;

        unsigned int vertex_seed = vertex_idx < nVertices ? pfCluster[vertex_idx].seedRHIdx() : 0;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const unsigned int cc_rhf_global_offset =
              rep_cc_idx > 0 ? args[group * blockDim + rep_cc_idx].blockRHFOffset() + blockRHFShift : blockRHFShift;

          if (is_representative)
            outPFCluster[global_topo_idx].rhfracOffset() = cc_rhf_global_offset;

          const unsigned int cc_rhf_relative_offset = args[vertex_idx].ccRHFOffset();

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int rhf_end = rhf_begin + rhf_size;

          unsigned int dst_rhf_offset = cc_rhf_relative_offset + cc_rhf_global_offset;

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
                  update_params = iter_lane_idx < eff_w_extent &&
                                  (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
                  continue;
                }

                const unsigned int proc_rhf_size = is_master_lane ? (rhf_size - 1) : 0;  //exclude master lane itself..
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
                const warp::warp_mask_t coop_subgroup_mask =
                    warp::ballot_mask(acc, active_lanes_mask, is_coop_subgroup_lane);  //Note: it excludes master lane.
                // Erase corresponding bits in 'free_lanes_mask'
                free_lanes_mask &= ~coop_subgroup_mask;
                // Update parameters only for cooperative subgroup lanes (and the master lane)
                // if 'true' - do broadcast of iter. lane index and corresponding rechit offset from source (current iterative) lane:
                if (is_coop_subgroup_lane || is_master_lane)
                  proc_lane_idx =
                      warp::shfl_mask(acc, coop_subgroup_mask | iter_lane_mask, iter_lane_idx, iter_lane_idx, w_extent);

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

                const unsigned int tmp_proc_lane_idx =
                    warp::shfl_mask(acc, free_lanes_mask | swap_lanes_mask, swap_lane_idx, src_phys_lane_idx, w_extent);

                if (is_work_lane(free_lanes_mask, lane_idx) && src_log_lane_idx < swap_lanes_num)
                  proc_lane_idx = tmp_proc_lane_idx;
              }

              const warp::warp_mask_t nonvacant_lanes_mask =
                  warp::ballot_mask(acc, active_lanes_mask, proc_lane_idx != w_extent);

              if (is_work_lane(nonvacant_lanes_mask, lane_idx) == false)
                continue;

              const warp::warp_mask_t coop_group_mask = warp::match_any_mask(acc, nonvacant_lanes_mask, proc_lane_idx);

              const float proc_cc_idx = warp::shfl_mask(acc, coop_group_mask, global_topo_idx, proc_lane_idx, w_extent);

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
              outPFRecHitFracs[dst_rhfrac_idx].pfcIdx() = global_topo_idx;
              ++dst_rhfrac_idx;
            }
          }
        }

        const bool is_block_local_represantative = (group * blockDim < rep_idx) || ((group + 1) * blockDim < rep_idx);

        if (is_isolated_root)
          args[global_topo_idx].ccEnergySeed() = static_cast<uint64_t>(vertex_seed);

        auto compFn = [] ALPAKA_FN_ACC(const float a, const float b) -> float { return a > b ? a : b; };

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          if (!is_block_local_represantative || is_isolated_root) {
            continue;
          }

          const warp::warp_mask_t rep_mask = alpaka::warp::activemask(acc);

          const unsigned int lane_idx = idx.local % w_extent;

          const float energy = pfRecHit[vertex_seed].energy();

          const warp::warp_mask_t subcomponent_mask = warp::match_any_mask(acc, rep_mask, rep_idx);

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

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          if (is_isolated_root)
            continue;

          uint64_t energy_seed = is_representative ? cc_energy_seed[rep_cc_idx] : 0;

          if (is_block_local_represantative == false) {
            const unsigned int lane_idx = idx.local % w_extent;

            const warp::warp_mask_t active_lane_mask = alpaka::warp::activemask(acc);

            const warp::warp_mask_t subcomponent_mask = warp::match_any_mask(acc, active_lane_mask, rep_idx);

            const float energy = pfRecHit[vertex_seed].energy();

            const float max_energy = warp_sparse_reduce(acc, subcomponent_mask, lane_idx, energy, compFn);

            const warp::warp_mask_t max_energy_lanes_mask =
                warp::ballot_mask(acc, subcomponent_mask, max_energy == energy);

            const unsigned int max_energy_lane_idx = alpaka::ffs(acc, static_cast<int>(max_energy_lanes_mask)) - 1;

            const auto seed_max = warp::shfl_mask(acc, subcomponent_mask, vertex_seed, max_energy_lane_idx, w_extent);
            const auto energy_max = warp::shfl_mask(acc, subcomponent_mask, max_energy, max_energy_lane_idx, w_extent);

            if (is_ls1b_idx<Acc1D>(subcomponent_mask, lane_idx)) {
              uint32_t e_uint = std::bit_cast<unsigned int>(energy_max);
              energy_seed = (static_cast<uint64_t>(e_uint) << 32) | static_cast<uint64_t>(seed_max);
            }
          }
          // Energies are assumed non-negative; bit-cast uint ordering matches float ordering for atomicMax.
          if (energy_seed != 0) {
            alpaka::atomicMax(acc, &args[global_topo_idx].ccEnergySeed(), energy_seed);
          }
        }
      }
    }
  };

  class ECLCCLoadSeedsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFClusterDeviceCollection::View outPFCluster,
                                  reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View args) const {
      const unsigned int nComponents = outPFCluster.nSeeds();

      for (auto global_topo_idx : ::cms::alpakatools::uniform_elements(acc, nComponents)) {
        uint64_t energy_seed_packed = args[global_topo_idx].ccEnergySeed();
        uint32_t seed_to_store = static_cast<uint32_t>(energy_seed_packed & 0xFFFFFFFF);

        outPFCluster[global_topo_idx].seedRHIdx() = seed_to_store;
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
