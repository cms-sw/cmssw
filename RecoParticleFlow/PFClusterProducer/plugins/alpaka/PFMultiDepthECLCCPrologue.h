#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologue_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"

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

  template <unsigned int max_w_items = 32, bool multi_block = false>
  class ECLCCPrologueKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();
      static_assert(max_w_items <= 32, "ECLCCPrologueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      if constexpr (std::is_same_v<Device, alpaka::DevCpu> ||
                    std::is_same_v<alpaka::AccToTag<Acc1D>, alpaka::TagGpuHipRt> || multi_block) {
        if (::cms::alpakatools::once_per_grid(acc)) {
          unsigned int store_idx = 0;

          for (unsigned int dst_idx = 0; dst_idx < nVertices; dst_idx++) {
            pfClusteringEdgeVars[dst_idx].mdpf_adjacencyIndex() = store_idx;

            const unsigned int base_neigh_idx = pfClusteringCCLabels[dst_idx].mdpf_topoId();

            if (dst_idx != base_neigh_idx)
              pfClusteringEdgeVars[store_idx++].mdpf_adjacencyList() = base_neigh_idx;

            for (unsigned int iter_idx = 0; iter_idx < nVertices; iter_idx++) {
              if (iter_idx == dst_idx)
                continue;

              const unsigned int neigh_idx = pfClusteringCCLabels[iter_idx].mdpf_topoId();

              if (neigh_idx == dst_idx)
                pfClusteringEdgeVars[store_idx++].mdpf_adjacencyList() = iter_idx;
            }
          }
          pfClusteringEdgeVars[nVertices].mdpf_adjacencyIndex() = store_idx;
        }
        return;
      } else {
        const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

        const unsigned int w_items = alpaka::math::min(acc, (blockDim + (w_extent - 1)) / w_extent, max_w_items);

        // nlist_offsets plays two roles:
        // -- neighbor count per vertex during counting,
        // -- CSR begin-offset per vertex after prefix sums.
        auto& nlist_offsets(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));

        // Temporary CSR storage: adjacency_list[0..tot_nnz) built in shared and then copied to global.
        // Note: adjacency_list capacity is 2*nVertices (worst-case assumption for this graph model).
        auto& adjacency_list(alpaka::declareSharedVar<unsigned int[2 * max_w_items * w_extent], __COUNTER__>(acc));

        auto& adjacency_idx(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));

        // Per-warp total offsets used for coarse-grained scan across warps.
        auto& subdomain_offsets(alpaka::declareSharedVar<unsigned int[max_w_items + 1], __COUNTER__>(acc));

        unsigned int dst_vertex_idx = 0;
        unsigned int src_vertex_idx = 0;  //base (local base) neigbor index

        for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
          // This kernel is intended to run with a single block for the full graph.
          // (If multi-block support is needed, the CSR construction could be made block-partition aware.)
          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            if (idx.global >= nVertices) {
              nlist_offsets[idx.local] = 0;
              adjacency_idx[idx.local] = 0;
              continue;
            }
            dst_vertex_idx = idx.global;

            src_vertex_idx = pfClusteringCCLabels[dst_vertex_idx].mdpf_topoId();

            // Init offset array with zero if locally isolated (self-connected) otherwise 1:
            nlist_offsets[idx.local] = dst_vertex_idx == src_vertex_idx ? 0 : 1;

            adjacency_idx[idx.local] = dst_vertex_idx == src_vertex_idx ? 0 : 1;

            if (idx.local <= max_w_items)
              subdomain_offsets[idx.local] = 0;
          }
          alpaka::syncBlockThreads(acc);
          // Next, identify:
          //  - inner (intra-warp) neighbors: vertices within the same warp that share the same base_neighbor
          //  - outer (inter-warp) neighbors: vertices in other warps that point to a base_neighbor in this warp
          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
            // Skip inactive lanes, from this point all warp-level operations must be done with active_lanes_mask
            // or derived mask
            if (idx.global >= nVertices)
              continue;

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;
            // Lane self-mask (bit corresponding to this lane).
            const warp::warp_mask_t lane_mask = get_lane_mask(lane_idx);
            // Identify warp-local domain range:
            const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
            const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

            // Check, whether this vertex is warp-local
            // For example (assume a toy model with warp-size equal to 4) we have the following list of neighbors
            // Neighbors = [2,2,7,0],[3,5,3,7],... that means that target vertex 0 connected to the source vertex 2 (denote as directed edge (0,2))
            // and so on, that is, (1,2),(2,7),(3,0),(4,3),(5,5),(6,3) and (7,7). Note that vertex 5 is isolated (no neighbors), while vertex 7
            // has neigbor vertex 2. Now we have 2 warps, namely [2,2,7,0] and [3,5,3,7], where the first one has 3 local edges, that is, (0,2),(1,2),(3,0)
            // and one inter-warp edge (2,7). The second one contains inter-warp edges only : (4,3) and (6,3)
            // In our toy model w_extent = 4, warp_idx = 0,1 and
            // for warp_idx = 0 : warp_low_boundary = 0, warp_high_boundary = 4 (excluded)
            // for warp_idx = 1 : warp_low_boundary = 4, warp_high_boundary = 8 (excluded)
            const bool is_warp_local_src_idx =
                (src_vertex_idx >= warp_low_boundary && src_vertex_idx < warp_high_boundary);
            // Assign warp lane index to each source vertex, for example, in the toy model vertex 7 is assign to lane 3 (as it can be seen)
            const int warp_local_src_vertex_lane_idx =
                is_warp_local_src_idx ? static_cast<int>(src_vertex_idx % w_extent) : -1;
            // and make corresponding lane mask:
            const warp::warp_mask_t warp_local_src_vertex_lane_mask =
                is_warp_local_src_idx ? get_lane_mask(warp_local_src_vertex_lane_idx) : 0;
            // We need to exclude all self-connections, like vertices 5 and 7 in the toy model
            // WARNING: neigh_num counts only directed neighbors. For vertex 7 neigh_num = 0 while it does have a neighbor via directed
            // edge (2,7)
            const bool is_self_connected = (src_vertex_idx == dst_vertex_idx);
            //unsigned int neigh_num = !is_self_connected ? 1 : 0;
            // Create a local adjacency list (in fact, just masking proper vertices), for a given linked source vertex
            // For the toy model one gets (warp 0): control_mask = 0x0011 for lane 0, control_mask =0x0011 for lane 1,
            // control_mask =0x0010 for lane 2 and control_mask =0x0001 for lane 3
            // For warp 1: one gets control_mask = 0x0101 for lane 4 and same for lane 6 etc.
            // Here we group lanes by identical src_vertex_idx using match_any. The resulting control_mask selects all lanes
            // that point to the same src_vertex_idx (within the active lanes).
            warp::warp_mask_t control_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);
            // Find out representative by lowest index (note that mask will select at least one lane, the very lane that
            // contains vertex):
            // For instance , for warp 0 lanes 0 and 1 are have same control mask, 0x0011, so we choose lane 0 is a local "represntative" (lane with the lowest id)
            // Note that if a vertex is locally or globally isolated (i.e, connected to itself), then it will always represent itself (even though it may
            // have higher index), i.e., if is_self_connected == true then rep_lane_idx = lane_id :
            // we choose a single representative lane for this group.
            // If the source vertex is warp-local, prefer the lane that actually owns src_vertex_idx;
            // otherwise pick the lowest lane in control_mask.
            const unsigned int rep_lane_idx =
                get_ls1b_idx(acc,
                             ((control_mask & warp_local_src_vertex_lane_mask) != 0) ? warp_local_src_vertex_lane_mask
                                                                                     : control_mask);
            // Exclude self-links (v -> v): they are sentinels for isolated vertices and must not appear as edges.
            if (is_self_connected)
              control_mask = control_mask ^ lane_mask;  // clear representative's own bit
            // Only the representative writes the group mask.
            // For warp-local sources we store the group under the 'source vertex index' so that the source vertex
            // can later account for inbound edges from same-warp destinations.
            // For non-local sources we keep the group under the destination index as 'outbound inter-warp' bookkeeping.

            if (lane_idx == rep_lane_idx) {
              const unsigned int neigh_num = alpaka::popcount(acc, control_mask);
              if (is_warp_local_src_idx) {  //i.e, src_vertex_idx is warp-local.
                adjacency_idx[src_vertex_idx] += neigh_num;
              }
              alpaka::atomicAdd(acc, &nlist_offsets[src_vertex_idx], neigh_num, alpaka::hierarchy::Threads{});
            }
          }

          alpaka::syncBlockThreads(acc);

          unsigned int cached_local_warp_offset = 0;

          // For the second stage, we compute local offsets for each lane (vertex) in the warp:
          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            const auto warp_idx = idx.local / w_extent;
            const auto lane_idx = idx.local % w_extent;

            const unsigned int nnz = nlist_offsets[idx.local];  //note that nnz = 0 for idx.local >= nVertices

            //WARNING: unlike a standard exclusive scan, where lane 0 gets 0 (that carries no useful information),
            //         our lane 0 stores total nnz:
            // - lanes 1..(w_extent-1) receive the exclusive prefix sum (CSR offsets within the warp),
            // - lane 0 receives the total sum over the warp (used as the per-warp NNZ aggregate).
            const auto local_warp_offset = warp_exclusive_sum(acc, nnz, lane_idx);
            // First, store total warp-local nnz into shared mem buffer for the lane id = 0:
            // local_warp_offset is the per-lane CSR offset within the warp (custom exclusive prefix sum of nnz).
            if (lane_idx == 0)
              subdomain_offsets[warp_idx] = local_warp_offset;
            // Second, store total local offsets into the private register:
            cached_local_warp_offset = lane_idx > 0 ? local_warp_offset : 0;
          }

          alpaka::syncBlockThreads(acc);

          // Construct coarse-grained offset (offsets for each warp):
          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            const auto warp_idx = idx.local / w_extent;

            if (warp_idx != 0)
              continue;
            // Create the full warp mask (all lanes will vote):
            const auto lane_idx = idx.local % w_extent;

            const auto local_warp_nnz = lane_idx < w_items ? subdomain_offsets[lane_idx] : 0;

            const auto block_local_warp_offset = warp_exclusive_sum(acc, local_warp_nnz, lane_idx);

            if (lane_idx < w_items)
              subdomain_offsets[lane_idx] = block_local_warp_offset;  //NOTE: lane 0 get total (block) nnz
          }

          alpaka::syncBlockThreads(acc);

          unsigned block_nnz = 0;

          // Now we assemble global offsets for each vertex:
          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nVertices);

            if (idx.global >= nVertices)
              continue;

            block_nnz = subdomain_offsets[0];

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;

            const unsigned lane_offset = cached_local_warp_offset;  // 0 for lane_idx = 0
            // Broadcast warp offset from lane 0 (for warp 0 it's just 0):
            const unsigned shift = warp_idx != 0 ? subdomain_offsets[warp_idx] : 0;

            const unsigned block_offset = lane_offset + shift;

            // Store final offsets in shared memory:
            adjacency_idx[idx.local] += block_offset;

            // Identify warp-local domain range:
            const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
            const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

            // Store block-level offsets in the  buffer:
            pfClusteringEdgeVars[dst_vertex_idx].mdpf_adjacencyIndex() = block_offset;

            unsigned int block_adjacency_pos = block_offset;

            if (src_vertex_idx != dst_vertex_idx)  // exclude self connection
              adjacency_list[block_adjacency_pos++] = src_vertex_idx;

            const bool is_warp_local_src_vertex_idx =
                (src_vertex_idx >= warp_low_boundary && src_vertex_idx < warp_high_boundary);

            const unsigned int warp_local_src_vertex_lane_idx =
                is_warp_local_src_vertex_idx ? src_vertex_idx % w_extent : lane_idx;

            // NOTE: the source lane does not know about warp_local_neigh_mask!
            const unsigned int src_block_adjacency_pos =
                warp::shfl_mask(acc, active_lanes_mask, block_adjacency_pos, warp_local_src_vertex_lane_idx, w_extent);
            // (exclude self-connection):
            const warp::warp_mask_t coop_mask = warp::ballot_mask(
                acc, active_lanes_mask, is_warp_local_src_vertex_idx && src_vertex_idx != dst_vertex_idx);

            if (is_work_lane(coop_mask, lane_idx, w_extent)) {
              const warp::warp_mask_t warp_local_neigh_mask = warp::match_any_mask(acc, coop_mask, src_vertex_idx);

              if (is_work_lane(warp_local_neigh_mask, lane_idx, w_extent)) {
                const unsigned int logical_lane_idx = get_logical_lane_idx(acc, warp_local_neigh_mask, lane_idx);

                adjacency_list[src_block_adjacency_pos + logical_lane_idx] = dst_vertex_idx;
              }
            }
          }
          alpaka::syncBlockThreads(acc);

          // Store tot_nnz in the global buffer:
          if (::cms::alpakatools::once_per_block(acc)) {
            pfClusteringEdgeVars[nVertices].mdpf_adjacencyIndex() = block_nnz;
          }

          for (auto idx : ::cms::alpakatools::uniform_group_elements(
                   acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.global < nVertices);

            if (idx.global >= nVertices)
              continue;

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;
            //
            const unsigned int warp_low_boundary = (warp_idx + 0) * w_extent + group * blockDim;
            const unsigned int warp_high_boundary = (warp_idx + 1) * w_extent + group * blockDim;

            const bool is_nonlocal_src_vertex_idx =
                (src_vertex_idx < warp_low_boundary || src_vertex_idx >= warp_high_boundary);

            // Detect all lanes that have external (inter-warp) neighbors
            const warp::warp_mask_t outer_totneighs_mask =
                warp::ballot_mask(acc, active_lanes_mask, is_nonlocal_src_vertex_idx);
            // Compute neighebor mask (internal or external):
            const warp::warp_mask_t totneighs_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);
            // Filter out internal neighbors (and clear non-representative lane bits):
            if (is_work_lane(outer_totneighs_mask, lane_idx, w_extent)) {
              const warp::warp_mask_t outer_neigh_mask = totneighs_mask & outer_totneighs_mask;

              const unsigned int rep_lane_idx = get_ls1b_idx(acc, outer_neigh_mask);

              const unsigned int nnz = static_cast<unsigned int>(alpaka::popcount(acc, outer_neigh_mask));

              const unsigned int block_local_src_vertex_idx = src_vertex_idx;

              const unsigned int block_adjacency_pos =
                  rep_lane_idx == lane_idx
                      ? alpaka::atomicAdd(
                            acc, &adjacency_idx[block_local_src_vertex_idx], nnz, alpaka::hierarchy::Threads{})
                      : 2 * nVertices;

              const unsigned int src_adjacency_pos =
                  warp::shfl_mask(acc, outer_totneighs_mask, block_adjacency_pos, rep_lane_idx, w_extent);

              const unsigned int logical_lane_idx = get_logical_lane_idx(acc, outer_neigh_mask, lane_idx);

              adjacency_list[src_adjacency_pos + logical_lane_idx] = dst_vertex_idx;
            }
          }

          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            for (unsigned int i = idx.local; i < block_nnz; i += nVertices) {
              pfClusteringEdgeVars[i].mdpf_adjacencyList() = adjacency_list[i];
            }
          }
        }  //groups
      }  // end of solo-block branch
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
