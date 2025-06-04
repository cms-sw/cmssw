#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCPrologue_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusterizerHelper.h"

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

  using namespace cms::alpakatools;
  using namespace reco::pfClustering;

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
 * It operates at warp level, efficiently detecting both intra-warp and inter-warp
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
 * - Neighbor detection: ballot, match_any_mask, mask manipulation.
 * - Warp-local prefix sum (exclusive sum) for offset computation.
 * - Intra-warp and inter-warp neighbor management.
 * - Atomic updates for global neighbor list in external connections.
 * 
 * - Only thread group `group == 0` processes the graph (single block), but this can be
 *   extended for true multi-block excution. 
 * - Shared memory scratch buffers are extensively used for intermediate neighbor masks.
 * 
 * - Care must be taken with shared memory sizing relative to `max_w_items * max_w_extent`.
 *
 */

  template <unsigned int max_w_items = 32>
  class ECLCCPrologueKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars,
        const reco::PFMultiDepthClusteringVarsDeviceCollection::ConstView pfClusteringVars) const {
      static_assert(max_w_items <= 32,
                    "ECLCCPrologueKernel: Maximum number of supported warps per block is 32, "
                    "assuming one warp per 32 threads.");
      constexpr unsigned int max_w_extent = 32;
      //
      const unsigned int nVertices = pfClusteringVars.size();
      //
      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];  //
      //
      const unsigned int w_extent = alpaka::warp::getSize(acc);
      const unsigned int w_items = alpaka::math::min(acc, nBlocks / w_extent, max_w_items);
      //
      auto& outer_neigh_masks(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      auto& inner_neigh_masks(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      //
      auto& base_neighbor(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      // Neighbor list offset
      auto& nlist_offsets(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<unsigned int, (max_w_items * max_w_extent + 1)>,
                                   __COUNTER__>(acc));
      //
      auto& adjacency_list(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, 2 * max_w_items * max_w_extent>, __COUNTER__>(
              acc));
      // Alias:
      auto& adjacency_idx = inner_neigh_masks;
      // Subdomain offsets:
      auto& subdomain_offsets(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<unsigned int, max_w_items>, __COUNTER__>(acc));

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        // Only single block is active:
        if (group != 0)
          continue;
        // First, we need to initialize shared_buffers:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          // Reset shared memory buffers to zero:
          nlist_offsets[idx.local] = 0;

          inner_neigh_masks[idx.local] = 0x0;

          outer_neigh_masks[idx.local] = 0x0;
          //
          if (idx.local < max_w_items)
            subdomain_offsets[idx.local] = 0;
          if (idx.local == 0)
            nlist_offsets[nVertices] = 0;
        }
        //
        alpaka::syncBlockThreads(acc);
        // Next, we need to find out internal (warp-local) and external (intra-warp) neighbors:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes, from this point all warp-level operations must be done with active_lanes_mask
          // or derived mask
          if (idx.local >= nVertices)
            continue;

          // we assume here that idx.local and idx.global have same range:
          const unsigned int dst_vertex_idx = idx.local;
          //
          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;
          // Usefull lane self-mask:
          const unsigned int lane_mask = 1 << lane_idx;
          // Identify warp-local domain range:
          const unsigned int src_vertex_low_idx = (warp_idx + 0) * w_extent;
          const unsigned int src_vertex_high_idx = (warp_idx + 1) * w_extent;
          //
          const unsigned int src_vertex_idx = pfClusteringVars[idx.global].mdpf_topoId();
          // Store source vertex index (direct neighbor) into the shared memory buffer:
          base_neighbor[idx.local] = src_vertex_idx;
          // Check, whether this vertex is warp-local
          // For example (assume a toy model with warp-size equal to 4) we have the following list of neighbors
          // Neighbors = [2,2,7,0,3,5,3,7], that means that target vertex 0 connected to the source vertex 2 (denote as directed edge (0,2))
          // and so on, that is, (1,2),(2,7),(3,0),(4,3),(5,5),(6,3) and (7,7). Note that vertex 5 is isolated (no neighbors), while vertex 7
          // has neigbor vertex 2. Now we have 2 warps, namely [2,2,7,0] and [3,5,3,7], where the first one has 3 local edges, that is, (0,2),(1,2),(3,0)
          // and one inter-warp edge (2,7). The second one contains inter-warp edges only : (4,3) and (6,3)
          // In our toy model w_extent = 4, warp_idx = 0,1 and
          // for warp_idx = 0 : src_vertex_low_idx = 0, src_vertex_high_idx = 4 (excluded)
          // for warp_idx = 1 : src_vertex_low_idx = 4, src_vertex_high_idx = 8 (excluded)
          const bool is_local_src_vertex_idx =
              (src_vertex_idx >= src_vertex_low_idx && src_vertex_idx < src_vertex_high_idx);
          // Assign warp lane index to each source vertex, for example, in the toy model vertex 7 is assign to lane 3 (as it can be seen)
          const int src_vertex_lane_idx = is_local_src_vertex_idx ? static_cast<int>(src_vertex_idx % w_extent) : -1;
          // and make corresponding lane mask:
          const unsigned int src_vertex_lane_mask = is_local_src_vertex_idx ? 1 << src_vertex_lane_idx : 0x0;
          // We need to exclude all self-connections, like vertices 5 and 7 in the toy model
          // WARNING: neigh_num counts only directed neighbors. For vertex 7 neigh_num = 0 while it does have a neighbor via directed
          // edge (2,7)
          const bool is_self_connected = (src_vertex_idx == dst_vertex_idx);
          unsigned neigh_num = is_self_connected == false ? 1 : 0;
          // Create a local adjacency list (in fact, just masking proper vertices), for a given linked source vertex
          // For the toy model one gets (warp 0): control_mask = 0x0011 for lane 0, control_mask =0x0011 for lane 1 etc
          // For warp 1: one gets control_mask = 0x0101 for lane 4 and same for lane 6
          unsigned int control_mask = warp::match_any_mask(acc, active_lanes_mask, src_vertex_idx);
          // Find out representative by lowest index (note that mask will select at least one lane, the very lane that
          // contains vertex, if the lane is selected by valid_vertices_mask, otherwise it will be set to -1):
          // For instance , for warp 0 lanes 0 and 1 are have same control mask, 0x0011, so we choose lane 0 is a local "represntative" (lane with the lowest id)
          // Note that if a vertex is locally or globally isolated (i.e, connected to itself), then it will always represent itself (even though it may
          // have higher index), i.e., if is_self_connected == true then rep_lane_idx = lane_id :
          const unsigned int rep_lane_idx =
              (((control_mask & src_vertex_lane_mask) != 0x0) ? src_vertex_lane_idx : get_ls1b_idx(acc, control_mask));
          // If a vertex represents itself, erase bit that corrsponds to the represntative vertex:
          if (is_self_connected)
            control_mask = control_mask ^ lane_mask;
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          if (lane_idx == rep_lane_idx) {
            if (is_local_src_vertex_idx)
              inner_neigh_masks[src_vertex_idx] = control_mask;  // internal (intra-warp) neighbors mask
            else
              outer_neigh_masks[dst_vertex_idx] = control_mask;  // external (inter-warp) neighbors mask
          }
          // We need to sync threads in the warp,
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Fetch other (possible) warp-local neighbors
          const unsigned int local_neigh_mask =
              inner_neigh_masks[dst_vertex_idx];  //dst_vertex_idx is in fact idx.local
          // update neighbor number:
          neigh_num += alpaka::popcount(acc, local_neigh_mask);
          //
          nlist_offsets[idx.local] = neigh_num;
        }

        alpaka::syncBlockThreads(acc);

        // Since we have collections of local neighbors represented as lane masks for each lane in a warp,
        // we need to construct offsets for the corresponding local adjacency lists. From a global perspective,
        // the goal is to build a (global) adjacency matrix in CSR format, which consists of an array of offsets
        // (defining the start of each adjacency list) and a flattened array of adjacency entries.
        // As the first step, we construct fine-grained (i.e., local) offsets below in 2 stages,
        // which will later be used to assemble the global offsets array.
        // For the first stage we compute number of local neighbors for each lane (vertex) in the warp:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          // we assume here that idx.local and idx.global have same range:
          const auto dst_vertex_idx = idx.local;
          //
          const unsigned int outer_neigh_mask = outer_neigh_masks[dst_vertex_idx];
          //
          const unsigned int outer_neigh_num = alpaka::popcount(acc, outer_neigh_mask);
          //
          if (outer_neigh_num == 0)
            continue;

          const unsigned int src_vertex_idx = base_neighbor[dst_vertex_idx];
          //
          alpaka::atomicAdd(acc, &nlist_offsets[src_vertex_idx], outer_neigh_num, alpaka::hierarchy::Threads{});
        }

        alpaka::syncBlockThreads(acc);

        // For the second stage, we compute local offsets for each lane (vertex) in the warp:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          // Create a full warp mask:
          const auto full_mask = alpaka::warp::ballot(acc, true);
          //
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          unsigned int nnz = nlist_offsets[idx.local];  //note that nnz = 0 for idx.local >= nVertices
          //
          const auto local_warp_offset = warp_exclusive_sum(acc, full_mask, nnz, lane_idx);
          // First, store total warp-local nnz into shared mem buffer for the lane id = 0:
          if (lane_idx == 0)
            subdomain_offsets[warp_idx] = local_warp_offset;
          // Second, store total local offsets into shared mem buffer:
          nlist_offsets[idx.local] = lane_idx > 0 ? local_warp_offset : 0;
        }

        alpaka::syncBlockThreads(acc);

        // Constract coarse-grained offset (offsets for each warp):
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const auto warp_idx = idx.local / w_extent;
          //
          if (warp_idx != 0)
            continue;
          // Create the full warp mask (all lanes will vote):
          const auto full_mask = alpaka::warp::ballot(acc, true);
          const auto lane_idx = idx.local % w_extent;
          //
          const auto local_warp_nnz = lane_idx < w_items ? subdomain_offsets[lane_idx] : 0;
          //
          const auto global_warp_offset = warp_exclusive_sum(acc, full_mask, local_warp_nnz, lane_idx);

          if (lane_idx < w_items)
            subdomain_offsets[lane_idx] = global_warp_offset;  //NOTE: lane 0 get total nnz
        }

        alpaka::syncBlockThreads(acc);

        // Now we assemble global offsets for each vertex:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          //
          if (idx.local >= nVertices)
            continue;
          //
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned warp_offset = lane_idx == 0 ? subdomain_offsets[warp_idx] : 0;
          const unsigned lane_offset = nlist_offsets[idx.local];  // 0 for lane_idx = 0
          // We just need to sync threads in the warp,
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Broadcast warp offset from lane 0 (for warp 0 it's just 0):
          const unsigned shift = warp_idx != 0 ? alpaka::warp::shfl(acc, warp_offset, 0, w_extent) : 0;
          //
          const auto global_offset = lane_offset + shift;
          // We just need to sync threads in the warp,
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Store final offsets in shared memory:
          nlist_offsets[idx.local] = global_offset;
          // Last entry for total NNZ:
          if (warp_idx == 0 && lane_idx == 0)
            nlist_offsets[nVertices] = warp_offset;
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          //
          if (idx.local >= nVertices)
            continue;
          //
          // Determin actual warp-level work dimension: it coincides with w_extent for all warps
          // except (potentially!) the last one:
          const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);

          const auto dst_vertex_idx = idx.local;
          //
          const auto warp_idx = idx.local / w_extent;
          // Get local coordinate:
          const unsigned begin = nlist_offsets[idx.local];
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          const unsigned int inner_neigh_mask = inner_neigh_masks[idx.local];
          //
          const unsigned int src_vertex_idx = base_neighbor[dst_vertex_idx];
          //
          unsigned int adjacency_pos = begin;
          //
          if (src_vertex_idx != dst_vertex_idx)
            adjacency_list[adjacency_pos++] = src_vertex_idx;

          for (unsigned lid = 0; lid < warp_work_extent; ++lid) {
            const auto target_lane_id = (inner_neigh_mask >> lid) & 1;
            if (target_lane_id != 0) {
              const unsigned int neigh_vertex_idx = lid + warp_idx * w_extent;
              adjacency_list[adjacency_pos++] = neigh_vertex_idx;
            }
          }

          warp::syncWarpThreads_mask(acc, active_lanes_mask);

          adjacency_idx[dst_vertex_idx] = adjacency_pos;
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          if (idx.local >= nVertices)
            continue;

          const auto dst_vertex_idx = idx.local;
          //
          const auto warp_idx = idx.local / w_extent;
          //
          // Get local coordinate:
          const unsigned int src_vertex_idx = base_neighbor[dst_vertex_idx];
          //
          const unsigned int outer_neigh_mask = outer_neigh_masks[idx.local];

          const auto nnz = alpaka::popcount(acc, outer_neigh_mask);

          if (nnz == 0)
            continue;  // skip inactive lanes

          unsigned adjacency_pos =
              alpaka::atomicAdd(acc, &adjacency_idx[src_vertex_idx], nnz, alpaka::hierarchy::Threads{});

          for (unsigned lid = 0; lid < w_extent; ++lid) {
            const auto target_lane_idx = (outer_neigh_mask >> lid) & 1;  // number of target lanes is equal to nnz
            if (target_lane_idx != 0) {
              const unsigned int neigh_vertex_idx = lid + warp_idx * w_extent;
              adjacency_list[adjacency_pos++] = neigh_vertex_idx;
            }
          }
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          if (idx.local >= nVertices)
            continue;
          //
          pfClusteringEdgeVars[idx.local].mdpf_adjacencyList() = adjacency_list[idx.local];
          pfClusteringEdgeVars[idx.local].mdpf_adjacencyIndex() = nlist_offsets[idx.local];
          //
          if (idx.local == 0)
            pfClusteringEdgeVars[nVertices].mdpf_adjacencyIndex() = nlist_offsets[nVertices];
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
