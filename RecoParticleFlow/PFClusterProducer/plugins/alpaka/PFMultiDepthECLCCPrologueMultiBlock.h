#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologueMultiBlock_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologueMultiBlock_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCPrologueArgsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

/**
 * @brief Warp-based construction of adjacency graph for multi-depth particle flow clusters.
 *
 * This header defines and implements an Alpaka kernel that builds the local neighbor structure
 * (adjacency lists) between particle flow clusters in preparation for 
 * connected components labeling algorithm (ECL-CC).
 * 
 * The kernel operates entirely at warp level, detecting both intra-warp and inter-warp neighbors
 * using masked ballot and shuffle operations. It produces a compressed sparse row (CSR) representation
 * of the cluster adjacency graph, suitable for efficient graph-based clustering algorithms.
 * 
 * Steps:
 * - Detect warp-local neighbors using ballots and match-any masks.
 * - Detect external (inter-warp) neighbors and handle them with atomic operations.
 * - Assemble per-warp and global prefix sums to build adjacency list offsets.
 * - Populate flattened adjacency list and offset arrays.
 * - Write adjacency structure into the device-side `PFMultiDepthClusteringEdgeVarsDeviceCollection`.
 *
 * - Warp-masked operations are used consistently to ensure efficiency and avoid divergence.
 * - Shared memory is used heavily for local neighbor data caching.
 * - Only a single thread group (`group == 0`) is active during execution.
 * - The output CSR structure (offsets + adjacency list) is used in subsequent ECL-CC graph processing.
 *
 * Additional notes:
 * - Shared memory consumption is proportional to the number of clusters and maximum warp size.
 *   Ensure device shared memory is sufficient for the intended cluster sizes.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::alpakatools;
  using namespace ::cms::alpakaintrinsics;

  class ECLCCComputeExternNeighsKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int w_extent = alpaka::warp::getSize(acc);

      if (::cms::alpakatools::once_per_grid(acc)) {
        args.blockCount() = 0;
      }

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int src_vertex_idx = pfClusteringCCLabels[idx.global].mdpf_topoId();
          const warp::warp_mask_t blk_nonlocal_mask =
              warp::ballot_mask(acc,
                                active_lanes_mask,
                                (src_vertex_idx < group * blockDim) || (src_vertex_idx >= (group + 1) * blockDim));
          // Note : from this point, a vertex cannot be self-connected.
          if (is_work_lane(blk_nonlocal_mask, lane_idx) == false)
            continue;

          const warp::warp_mask_t control_mask = warp::match_any_mask(acc, blk_nonlocal_mask, src_vertex_idx);

          if (is_ls1b_idx<Acc1D>(control_mask, lane_idx)) {
            const unsigned int extern_neigh_num = alpaka::popcount(acc, control_mask);
            alpaka::atomicAdd(acc, &args[src_vertex_idx].ccSize(), extern_neigh_num, alpaka::hierarchy::Threads{});
          }
        }
      }
    }
  };

  /**
 * @class ECLCCPrologueKernel
 * @brief Kernel for constructing the initial adjacency graph for ECL-CC clustering.
 * 
 * @tparam max_w_items Maximum number of warp-sized tiles processed by a single block,
 *                     Controls shared memory footprint and parallelism 
 *                     (currently default to 32, and must be <= 32).
 * The ECLCCPrologueKernel builds the compressed sparse row (CSR) representation
 * of the connectivity graph between particle flow clusters.
 * 
 * It operates at warp level, detecting both intra-warp and inter-warp
 * neighbors based on precomputed cluster links (e.g., from geometric matching).
 * 
 * The adjacency information is stored into the `PFMultiDepthClusteringEdgeVarsDeviceCollection`,
 * and consists of:
 * - `mdpf_adjacencyList()`: Flat array of neighbor indices.
 * - `mdpf_adjacencyIndex()`: CSR-style offset array per cluster.
 * 
 * The resulting adjacency graph is used as input for graph-based connected
 * components labeling (ECL-CC).
 *
 * @algorithm
 * - Neighbor detection using ballot, match_any_mask, mask manipulation.
 * - Warp-local prefix sum (exclusive sum) for offset computations.
 * - Intra-warp and inter-warp neighbor management.
 * - Atomic updates for global neighbor list in external connections.
 * 
 * - Only thread group `group == 0` processes the graph (single block), but this can be
 *   extended for true multi-block excution. 
 * - Shared memory scratch buffers are extensively used for intermediate neighbor masks.
 * 
 * - The algorithm assumes that each destination vertex is connected to just a single source vertex, 
 * - Same source can be linked to one or many destination vertices, or isolated. 
 * 
 * @param acc                  Alpaka accelerator instance.
 * @param pfClusteringEdgeVars Output view for per-edge clustering variables.
 * @param pfClusteringCCLabels Input view of connected-component labels.
 */

  template <unsigned int max_w_items = 32>
  class ECLCCPrologueComputeOffsetsKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();
      static_assert(max_w_items <= 32, "ECLCCPrologueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];

      const unsigned int w_items = alpaka::math::min(acc, (blockDim + (w_extent - 1)) / w_extent, max_w_items);

      // warp_local_nlist and block_local_nlist plays two roles:
      // -- neighbor count per vertex during counting,
      // -- CSR begin-offset per vertex after prefix sums.
      auto& warp_local_nlist(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));
      auto& block_local_nlist(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));

      // Per-warp total offsets used for coarse-grained scan across warps.
      auto& subdomain_offsets(alpaka::declareSharedVar<unsigned int[w_extent], __COUNTER__>(acc));

      unsigned int& isLastBlockDone = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      unsigned int dst_vertex_idx = nVertices;
      unsigned int src_vertex_idx = nVertices;

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        // This kernel is intended to run with a single block for the full graph.
        // (If multi-block support is needed, the CSR construction could be made block-partition aware.)
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          dst_vertex_idx = idx.global;

          src_vertex_idx = pfClusteringCCLabels[dst_vertex_idx].mdpf_topoId();

          // Init offset array with zero if locally isolated (self-connected) otherwise 1:
          warp_local_nlist[idx.local] = dst_vertex_idx == src_vertex_idx ? 0 : 1;
          block_local_nlist[idx.local] = 0;

          if (idx.local < max_w_items) {
            subdomain_offsets[idx.local] = 0;
          }
        }

        alpaka::syncBlockThreads(acc);
        // Next, identify:
        //  - inner (intra-warp) neighbors: vertices within the same warp that share the same base_neighbor
        //  - outer (inter-warp) neighbors: vertices in other warps that point to a base_neighbor in this warp
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);
          // from this point all warp-level operations must be done with active_lanes_mask
          // or derived mask
          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;
          // Lane self-mask (bit corresponding to this lane).
          const warp::warp_mask_t lane_mask = get_lane_mask(lane_idx);
          // Identify warp-local domain range:
          const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
          const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

          const bool is_warp_local_src_idx =
              (src_vertex_idx >= warp_low_boundary && src_vertex_idx < warp_high_boundary);
          const unsigned int warp_local_src_vertex_lane_idx =
              is_warp_local_src_idx ? src_vertex_idx % w_extent : w_extent;

          const warp::warp_mask_t warp_local_src_vertex_lane_mask = get_lane_mask(warp_local_src_vertex_lane_idx);
          const bool is_self_connected = (src_vertex_idx == dst_vertex_idx);  //!!!

          warp::warp_mask_t control_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);

          const unsigned int rep_lane_idx = get_ls1b_idx(
              acc,
              ((control_mask & warp_local_src_vertex_lane_mask) != 0) ? warp_local_src_vertex_lane_mask : control_mask);
          // Exclude self-links (v -> v): they are sentinels for isolated vertices and must not appear as edges.
          if (is_self_connected)
            control_mask &= ~lane_mask;  // clear representative's own bit

          if (lane_idx == rep_lane_idx) {
            const unsigned int neigh_num = alpaka::popcount(acc, control_mask);
            const unsigned int block_local_src_vertex_idx = src_vertex_idx % blockDim;
            if (is_warp_local_src_idx) {  //no race condition here:
              warp_local_nlist[block_local_src_vertex_idx] += neigh_num;
            } else if ((src_vertex_idx >= group * blockDim) && (src_vertex_idx < (group + 1) * blockDim)) {
              alpaka::atomicAdd(
                  acc, &block_local_nlist[block_local_src_vertex_idx], neigh_num, alpaka::hierarchy::Threads{});
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        const unsigned int block_external_nnz = (dst_vertex_idx < nVertices) ? args[dst_vertex_idx].ccSize() : 0;

        unsigned int cached_warp_local_offset = 0;

        // For the second stage, we compute local offsets for each lane (vertex) in the warp:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;

          const unsigned int block_internal_nnz = warp_local_nlist[idx.local] + block_local_nlist[idx.local];

          args[idx.global].blockInternCCSize() = block_internal_nnz;

          //WARNING: unlike a standard exclusive scan, where lane 0 gets 0 (that carries no useful information),
          //         our lane 0 stores total nnz:
          // - lanes 1..(w_extent-1) receive the exclusive prefix sum (CSR offsets within the warp),
          // - lane 0 receives the total sum over the warp (used as the per-warp NNZ aggregate).
          // Store total local offsets into the private register (but lane 0 gets total nnz):
          cached_warp_local_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, block_internal_nnz, lane_idx);
          // Store total warp-local nnz into shared mem buffer for the lane id = 0:
          // local_warp_offset is the per-lane CSR offset within the warp (custom exclusive prefix sum of nnz).
          if (lane_idx == 0) {
            subdomain_offsets[warp_idx] = cached_warp_local_offset;
            // reset local offsets:
            cached_warp_local_offset = 0;
          }
        }
        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const auto lane_idx = idx.local % w_extent;
          if (lane_idx >= w_items)
            continue;
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const auto nnz = subdomain_offsets[lane_idx];

          const auto cross_warp_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, nnz, lane_idx);

          subdomain_offsets[lane_idx] = cross_warp_offset;  //NOTE: lane 0 get total (block) nnz
        }
        alpaka::syncBlockThreads(acc);

        int store_offset = 0;

        // Now we assemble global offsets for each vertex:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const unsigned int warp_idx = idx.local / w_extent;

          // Broadcast warp offset from lane 0 (for warp 0 it's just 0):
          cached_warp_local_offset += (warp_idx != 0 ? subdomain_offsets[warp_idx] : 0);

          store_offset = idx.local == 0 ? subdomain_offsets[0] : cached_warp_local_offset;

          args[dst_vertex_idx].ccLocalOffset() = store_offset;
        }
        alpaka::syncBlockThreads(acc);

        // Now repeat for the block-external neighbors:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          if (idx.local < w_extent)
            subdomain_offsets[idx.local] = 0;
        }
        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;

          cached_warp_local_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, block_external_nnz, lane_idx);

          if (lane_idx == 0) {
            subdomain_offsets[warp_idx] = cached_warp_local_offset;
            // reset local offsets:
            cached_warp_local_offset = 0;
          }
        }
        alpaka::syncBlockThreads(acc);

        // Constract coarse-grained offset (offsets for each warp):
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, w_extent)) {
          const auto lane_idx = idx.local % w_extent;
          if (lane_idx >= w_items)
            continue;

          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const auto nnz = subdomain_offsets[lane_idx];

          const auto cross_warp_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, nnz, lane_idx);

          subdomain_offsets[lane_idx] = cross_warp_offset;  //NOTE: lane 0 get total (block) nnz
        }
        alpaka::syncBlockThreads(acc);

        // Now we assemble global offsets for each vertex:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const unsigned int warp_idx = idx.local / w_extent;

          // Broadcast warp offset from lane 0 (for warp 0 it's just 0):
          cached_warp_local_offset += (warp_idx != 0 ? subdomain_offsets[warp_idx] : 0);

          store_offset += idx.local == 0 ? subdomain_offsets[0] : cached_warp_local_offset;

          args[dst_vertex_idx].ccOffset() = store_offset;  // contains total block neigh number for idx.local = 0
        }
        alpaka::syncBlockThreads(acc);
        // Store tot_nnz in the global buffer (for single-block exec):
        if (::cms::alpakatools::once_per_block(acc)) {
          const unsigned int whichBlockDone =
              alpaka::atomicAdd(acc, &args.blockCount(), +1, alpaka::hierarchy::Threads{});

          alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

          isLastBlockDone = whichBlockDone == (nBlocks - 1) ? 1 : 0;
        }

        alpaka::syncBlockThreads(acc);

        if (isLastBlockDone == 1) {
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            // Create the full warp mask (all lanes will vote):
            const auto lane_idx = idx.local % w_extent;

            if (lane_idx >= nBlocks)
              continue;

            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const auto load_idx = lane_idx * blockDim;

            const auto global_nnz = args[load_idx].ccOffset();

            const auto global_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, global_nnz, lane_idx);

            args[load_idx].ccOffset() = lane_idx > 0 ? global_offset : 0;

            if (lane_idx == 0)
              args.size() = global_offset;
          }
        }
      }
    }
  };

  template <unsigned int max_w_items = 32, bool is_cooperative = true>
  class ECLCCFinalizePrologueKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars,
        reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();
      static_assert(max_w_items <= 32, "ECLCCPrologueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

      if (::cms::alpakatools::once_per_grid(acc)) {
        pfClusteringEdgeVars[nVertices].mdpf_adjacencyIndex() = args.size();
      }

      auto& adjacency_pos(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));

      // Temporary CSR storage: adjacency_list[0..tot_nnz) built in shared and then copied to global.
      // Note: adjacency_list capacity is 2*nVertices (worst-case assumption for this graph model).
      auto& adjacency_list(alpaka::declareSharedVar<unsigned int[2 * max_w_items * w_extent], __COUNTER__>(acc));

      unsigned int& base = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      unsigned int dst_vertex_idx = nVertices;
      unsigned int src_vertex_idx = nVertices;

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        // This kernel is intended to run with a single block for the full graph.
        // (If multi-block support is needed, the CSR construction could be made block-partition aware.)
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          dst_vertex_idx = idx.global;
          src_vertex_idx = pfClusteringCCLabels[dst_vertex_idx].mdpf_topoId();

          adjacency_list[idx.local] = 0;
          adjacency_pos[idx.local] = 0;
        }

        const bool is_self_connected = dst_vertex_idx < nVertices ? (dst_vertex_idx == src_vertex_idx) : false;

        if (::cms::alpakatools::once_per_block(acc)) {
          base = group > 0 ? args[group * blockDim].ccOffset() : 0;
        }

        alpaka::syncBlockThreads(acc);

        const unsigned int local_offset = dst_vertex_idx < nVertices ? args[dst_vertex_idx].ccLocalOffset() : 0;

        unsigned int offset = 0;

        // Now we assemble global offsets for each vertex:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          offset = idx.local > 0 ? args[dst_vertex_idx].ccOffset() + base : base;

          args[dst_vertex_idx].ccOffset() = offset;

          pfClusteringEdgeVars[dst_vertex_idx].mdpf_adjacencyIndex() = offset;

          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;

          unsigned int current_offset = idx.local > 0 ? local_offset : 0;

          // Identify warp-local domain range:
          const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
          const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

          const bool is_warp_local_src_vertex_idx =
              (src_vertex_idx >= warp_low_boundary && src_vertex_idx < warp_high_boundary);

          if (is_self_connected == false) {  // exclude self connection
            adjacency_list[current_offset++] = src_vertex_idx;
          }

          // Store offsets in shared memory:
          adjacency_pos[idx.local] =
              current_offset;  //not that is current_offset is schifted by 1 for non-self connected nodes

          warp::syncWarpThreads_mask(acc, active_lanes_mask);

          const unsigned int warp_local_src_vertex_lane_idx =
              is_warp_local_src_vertex_idx ? src_vertex_idx % w_extent : lane_idx;

          // NOTE: the source lane does not know about warp_local_neigh_mask!
          const unsigned int src_block_adjacency_pos =
              warp::shfl_mask(acc, active_lanes_mask, current_offset, warp_local_src_vertex_lane_idx, w_extent);
          // (exclude self-connection):
          const warp::warp_mask_t coop_mask =
              warp::ballot_mask(acc, active_lanes_mask, is_warp_local_src_vertex_idx && !is_self_connected);

          if (is_work_lane(coop_mask, lane_idx)) {
            const warp::warp_mask_t warp_local_neigh_mask = warp::match_any_mask(acc, coop_mask, src_vertex_idx);

            if (is_work_lane(warp_local_neigh_mask, lane_idx)) {
              const unsigned int logical_lane_idx = get_logical_lane_idx(acc, warp_local_neigh_mask, lane_idx);

              adjacency_list[src_block_adjacency_pos + logical_lane_idx] = dst_vertex_idx;
              // no race condition since src_vertex_idx % blockDim is a warp-local index:
              if (logical_lane_idx == 0)
                adjacency_pos[src_vertex_idx % blockDim] +=
                    static_cast<unsigned int>(alpaka::popcount(acc, warp_local_neigh_mask));
            }
          }
        }
        alpaka::syncBlockThreads(acc);

        const unsigned int block_internal_nnz =
            dst_vertex_idx < nVertices ? args[dst_vertex_idx].blockInternCCSize() : 0;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
          const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

          const bool is_nonlocal_src_vertex_idx =
              (src_vertex_idx < warp_low_boundary || src_vertex_idx >= warp_high_boundary);

          const bool is_block_local_src_vertex_idx =
              (src_vertex_idx >= group * blockDim) && (src_vertex_idx < (group + 1) * blockDim);

          // Detect all lanes that have external (inter-warp) neighbors
          const warp::warp_mask_t outer_totneighs_mask =
              warp::ballot_mask(acc, active_lanes_mask, is_nonlocal_src_vertex_idx && is_block_local_src_vertex_idx);
          // Compute neighebor mask (internal or external):
          const warp::warp_mask_t totneighs_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);
          // Filter out internal neighbors (and clear non-representative lane bits):
          if (is_work_lane(outer_totneighs_mask, lane_idx)) {
            const warp::warp_mask_t outer_neigh_mask = totneighs_mask & outer_totneighs_mask;

            const unsigned int rep_lane_idx = get_ls1b_idx(acc, outer_neigh_mask);

            const unsigned int nnz = static_cast<unsigned int>(alpaka::popcount(acc, outer_neigh_mask));

            const unsigned int block_local_src_vertex_idx = src_vertex_idx % blockDim;

            const unsigned int block_adjacency_pos =
                rep_lane_idx == lane_idx
                    ? alpaka::atomicAdd(
                          acc, &adjacency_pos[block_local_src_vertex_idx], nnz, alpaka::hierarchy::Threads{})
                    : 2 * nVertices;

            const unsigned int src_adjacency_pos =
                warp::shfl_mask(acc, outer_totneighs_mask, block_adjacency_pos, rep_lane_idx, w_extent);

            const unsigned int logical_lane_idx = get_logical_lane_idx(acc, outer_neigh_mask, lane_idx);

            adjacency_list[src_adjacency_pos + logical_lane_idx] = dst_vertex_idx;
          }
        }
        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          if constexpr (is_cooperative) {
            const unsigned int eff_w_extent = alpaka::popcount(acc, active_lanes_mask);

            const unsigned int lane_idx = idx.local % w_extent;

            unsigned int intern_nnz = block_internal_nnz;
            // Define iteration parameters:
            unsigned int iter_lane_idx = 0;

            unsigned int dst_offset = offset;
            unsigned int src_offset = idx.local > 0 ? local_offset : 0;

            const warp::warp_mask_t nonvoid_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, intern_nnz > 0);

            while (iter_lane_idx < eff_w_extent) {
              // Identify lanes whose remaining PFRecHitFraction work is exactly one element.
              // These lanes cannot benefit from recruiting cooperative helpers and are handled by a fast path.
              // Note: pfrhf_size may have been reduced in a previous iteration due to leftover handling.
              const warp::warp_mask_t single_worklane_mask = warp::ballot_mask(acc, active_lanes_mask, intern_nnz == 1);

              // Create temporary per-iteration masks for cooperative scheduling:
              // -- how many vacant lanes in the warp, i.e., lanes available to become cooperators in this iteration ('free_lanes_mask')
              // -- how many reserved lanes in the warp ('swap_lanes_mask'); note that free_lanes_mask must also include reserved lanes
              warp::warp_mask_t free_lanes_mask = active_lanes_mask;
              warp::warp_mask_t swap_lanes_mask = static_cast<warp::warp_mask_t>(0);

              unsigned int swap_lane_idx{w_extent};  //assigned to some default value (it's outside of the warp range)
              unsigned int swap_lanes_num{0};        //no lanes in the warp keep iter lane idx for swap operation

              unsigned int proc_lane_idx{w_extent};
              unsigned int proc_dst_offset{dst_offset};
              unsigned int proc_src_offset{src_offset};

              bool update_params = true;

              warp::syncWarpThreads_mask(acc, active_lanes_mask);

              while (update_params) {
                if (is_work_lane(nonvoid_lanes_mask, iter_lane_idx) == false) {
                  iter_lane_idx += 1;
                  update_params = iter_lane_idx < eff_w_extent;
                  continue;
                }
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
                    proc_dst_offset = dst_offset;
                    proc_src_offset = src_offset;
                  }
                  iter_lane_idx += 1;
                  update_params = iter_lane_idx < eff_w_extent &&
                                  (static_cast<std::uint32_t>(alpaka::popcount(acc, free_lanes_mask)) > swap_lanes_num);
                  continue;
                }

                const unsigned int proc_intern_nnz =
                    is_master_lane ? (intern_nnz - 1) : 0;  //exclude master lane itself..
                // Broadcast worksize (all active lanes):
                const unsigned int iter_intern_nnz =
                    warp::shfl_mask(acc, active_lanes_mask, proc_intern_nnz, iter_lane_idx, w_extent);

                const unsigned int coop_subgroup_size = alpaka::math::min(acc, iter_intern_nnz, free_subgroup_size);

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
                if (iter_intern_nnz <= free_subgroup_size) {
                  iter_lane_idx += 1;
                  // if 'false', we  have to update a leftover work for the current master lane (will continue in the next iteration):
                } else if (is_master_lane) {
                  // update rechit fraction offset for the next iteration
                  proc_dst_offset = dst_offset;
                  dst_offset +=
                      coop_subgroup_size + 1;  //we need to take into account the master lane itself (hence "+1" here)
                  proc_src_offset = src_offset;
                  src_offset +=
                      coop_subgroup_size + 1;  //we need to take into account the master lane itself (hence "+1" here)
                  // compute remaining work size (that is a leftover rec hit fraction size)
                  intern_nnz = iter_intern_nnz - coop_subgroup_size;
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

              const unsigned int coop_dst_offset =
                  warp::shfl_mask(acc, coop_group_mask, proc_dst_offset, proc_lane_idx, w_extent);
              const unsigned int coop_src_offset =
                  warp::shfl_mask(acc, coop_group_mask, proc_src_offset, proc_lane_idx, w_extent);

              const auto log_lane_idx = get_logical_lane_idx(acc, coop_group_mask, lane_idx);
              const int dst_idx = log_lane_idx + coop_dst_offset;
              const int src_idx = log_lane_idx + coop_src_offset;

              pfClusteringEdgeVars[dst_idx].mdpf_adjacencyList() = adjacency_list[src_idx];
            }
          } else {
            const unsigned int local_shift = idx.local > 0 ? local_offset : 0;
            for (unsigned int j = 0; j < block_internal_nnz; j++) {
              pfClusteringEdgeVars[offset + j].mdpf_adjacencyList() = adjacency_list[j + local_shift];
            }
          }
        }
      }
    }
  };

  class ECLCCLoadCrossBlockNeighKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars,
        reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int w_extent = alpaka::warp::getSize(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int src_vertex_idx = pfClusteringCCLabels[idx.global].mdpf_topoId();
          const unsigned int dst_vertex_idx = idx.global;

          const bool is_non_block_local_src_vertex_idx =
              (src_vertex_idx < group * blockDim) || (src_vertex_idx >= (group + 1) * blockDim);

          // Detect all lanes that have external (inter-warp) neighbors
          const warp::warp_mask_t outer_totneighs_mask =
              warp::ballot_mask(acc, active_lanes_mask, is_non_block_local_src_vertex_idx);

          // Compute neighebor mask (internal or external):
          const warp::warp_mask_t totneighs_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);

          // Filter out internal neighbors (and clear non-representative lane bits):
          if (is_work_lane(outer_totneighs_mask, lane_idx)) {
            const warp::warp_mask_t outer_neigh_mask = totneighs_mask & outer_totneighs_mask;

            const unsigned int rep_lane_idx = get_ls1b_idx(acc, outer_neigh_mask);

            const unsigned int nnz_to_load = static_cast<unsigned int>(alpaka::popcount(acc, outer_neigh_mask));

            const unsigned int block_internal_nnz =
                rep_lane_idx == lane_idx ? args[src_vertex_idx].blockInternCCSize() : 0;

            const unsigned int global_adj_pos =
                rep_lane_idx == lane_idx
                    ? block_internal_nnz +
                          alpaka::atomicAdd(
                              acc, &args[src_vertex_idx].ccOffset(), nnz_to_load, alpaka::hierarchy::Threads{})
                    : 2 * nVertices;

            const unsigned int src_adjacency_pos =
                warp::shfl_mask(acc, outer_totneighs_mask, global_adj_pos, rep_lane_idx, w_extent);

            const unsigned int logical_lane_idx = get_logical_lane_idx(acc, outer_neigh_mask, lane_idx);

            pfClusteringEdgeVars[src_adjacency_pos + logical_lane_idx].mdpf_adjacencyList() = dst_vertex_idx;
          }
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
