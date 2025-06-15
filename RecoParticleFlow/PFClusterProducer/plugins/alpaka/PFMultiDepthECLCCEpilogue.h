#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
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
 * - Masked warp-scope reductions to ensure efficient and divergence-free operations.
 *
 * Key outputs:
 * - mdpf_component()       : representative vertex index per cluster.
 * - mdpf_componentEnergy() : total rechit energy for each component.
 * - mdpf_componentIndex()  : compressed component index for final sorting.
 *
 * - Warp-masked ballot, shuffle, and scan operations are used throughout.
 * - Shared memory usage depends on max_w_items ensure adequate resource sizing.
 * - Only a single block (group == 0) is active during execution.
 * 
 * Ensure consistency between Prologue (adjacency construction) and Epilogue (component labeling) stages.
 *
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  /**
 * @class ECLCCEpilogueKernel
 * @brief Finalizes cluster component information after ECL-CC labeling.
 *
 * The ECLCCEpilogueKernel aggregates information about particle flow clusters
 * after they have been linked into connected components by the ECL-CC algorithm.
 * 
 * Responsibilities:
 * - Calculate total rechit energy per component.
 * - Map cluster vertices to their connected component representatives.
 * - Assign compressed component indices for further processing.
 * 
 * - Warp-masked operations are used throughout to eliminate divergence.
 * - Component aggregation and rechit assignment use warp-level masked scans.
 * 
 * @tparam max_w_items Maximum number of warp tiles processed per block () controls shared memory footprint ).
 *
 */

  template <unsigned int max_w_items = 32>
  class ECLCCEpilogueKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::PFClusterDeviceCollection::View outPFCluster,
                                  reco::PFRecHitFractionDeviceCollection::View outPFRecHitFracs,
                                  const reco::PFMultiDepthClusteringVarsDeviceCollection::ConstView pfClusteringVars,
                                  const reco::PFClusterDeviceCollection::ConstView pfCluster,
                                  const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
                                  const reco::PFRecHitDeviceCollection::ConstView pfRecHit) const {
      static_assert(max_w_items <= 32,
                    "ECLCCEpilogueKernel: Maximum number of supported warps per block is 32, "
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
      auto& intern_connected_comp_masks(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      auto& extern_connected_comp_masks(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      //
      auto& component_roots(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent + 1>, __COUNTER__>(
              acc));
      auto& connected_comp_buffer(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent + 1>, __COUNTER__>(
              acc));
      //
      auto& connected_comp_offsets = connected_comp_buffer;
      auto& connected_comp_sizes = connected_comp_buffer;
      //
      //auto& comp_offsets = component_roots;
      auto& topol_comp_offsets(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent + 1>, __COUNTER__>(
              acc));

      auto& component_cluster_seeds(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<float, max_w_items * max_w_extent>, __COUNTER__>(acc));
      auto& component_cluster_energies(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<float, max_w_items * max_w_extent>, __COUNTER__>(acc));

      auto& component_vertex_rhf_offsets(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      auto& component_vertex_rhf_sizes(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<int, max_w_items * max_w_extent>, __COUNTER__>(acc));
      //
      auto& connected_comp_vertices = component_vertex_rhf_offsets;
      auto& connected_comp_pos = component_vertex_rhf_sizes;
      //
      auto& subdomain_offsets(
          alpaka::declareSharedVar<::cms::alpakatools::VecArray<unsigned int, max_w_items>, __COUNTER__>(acc));
      //
      // Setup all shared mem buffers:
      //
      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {  //loop over thread blocks
        // Skip inactive groups:
        if (group != 0)
          continue;
        // Init shared_buffer
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const auto warp_idx = idx.local / w_extent;
          // Reset shared memory buffers to zero:
          component_roots[idx.local] = -1;
          connected_comp_buffer[idx.local] = 0;
          topol_comp_offsets[idx.local] = 0;
          //
          intern_connected_comp_masks[idx.local] = 0x0;
          extern_connected_comp_masks[idx.local] = 0x0;
          //
          component_cluster_seeds[idx.local] = 0.0f;
          component_cluster_energies[idx.local] = 0.0f;
          //
          component_vertex_rhf_offsets[idx.local] = 0;
          component_vertex_rhf_sizes[idx.local] = 0;
          //
          if (warp_idx == 0)
            subdomain_offsets[idx.local] = 0;
        }
        //
        alpaka::syncBlockThreads(acc);
        // Identify all neigbors:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          const auto vertex_idx = idx.local;
          //
          //const auto warp_idx   = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned int rep_idx = pfClusteringVars[vertex_idx].mdpf_topoId();
          //
          component_roots[vertex_idx] = rep_idx;
          //
          component_cluster_seeds[idx.local] = pfCluster[vertex_idx].seedRHIdx();  //!!!
          //
          component_vertex_rhf_offsets[idx.local] = pfCluster[vertex_idx].rhfracOffset();
          component_vertex_rhf_sizes[idx.local] = pfCluster[vertex_idx].rhfracSize();

          // Find out as to whether the current lane holds the representative vertex:
          const bool is_warp_local_representative = vertex_idx == rep_idx;
          // Find out how many vertices in the warp connected to a given representative:
          warp::syncWarpThreads_mask(acc, active_lanes_mask);

          unsigned int component_mask = warp::match_any_mask(acc, active_lanes_mask, rep_idx);
          // Compute number of such vertices. Note that intern_component_size is always > 0
          // since each vertex locally at least self-connected, that is, it can be locally isolated.
          const unsigned int component_size = alpaka::popcount(acc, component_mask);
          // Define a master lane for each sub-component, note that the lane which holds the root is
          // always selected as master, otherwise choose the lane with the lowest index.
          // Note that, by construction, if a vertex happened to be the local reprentaitve, it has always the lowest
          // lane index.
          const unsigned int master_lane_idx = get_ls1b_idx(acc, component_mask);
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Store internal/external component masks in the shared memory
          if (master_lane_idx == lane_idx) {
            if (is_warp_local_representative) {
              // no race condition : each lane works with unique representative (or idle)
              connected_comp_sizes[rep_idx] = component_size;
              intern_connected_comp_masks[rep_idx] = component_mask;
            } else {
              // no race condition : by construction each lane load to a unique location
              extern_connected_comp_masks[vertex_idx] = component_mask;
            }
          }
        }
        //
        alpaka::syncBlockThreads(acc);
        //

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          const auto vertex_idx = idx.local;
          //
          const unsigned int component_mask = extern_connected_comp_masks[vertex_idx];
          //
          const int component_size = alpaka::popcount(acc, component_mask);
          //
          if (component_size == 0)
            continue;
          //
          const unsigned int rep_idx = component_roots[vertex_idx];
          alpaka::atomicAdd(acc, &connected_comp_sizes[rep_idx], component_size, alpaka::hierarchy::Threads{});
        }
        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const auto active_lanes_mask = alpaka::warp::ballot(acc, true);
          //
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned int component_size = connected_comp_sizes[idx.local];
          // Note that component_size = 0 for idx.local >= nVertices and connected child vertices,
          // only represntatives hold a non-zero value.
          const unsigned int valid_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, component_size > 0);
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // Warp-uniform operation. Note that we exclude a trivial case
          // (when valid_lanes_mask = 0x0, that is, when all vertices processed by a warp are connected to external representatives)
          //const auto  local_warp_offset = valid_lanes_mask != 0x0 ? warp_exclusive_sum(acc, active_lanes_mask, component_size, lane_idx) : 0;
          const auto local_warp_offset = warp_exclusive_sum(acc, valid_lanes_mask, component_size, lane_idx);
          // Store warp offsets in a separate buffer:
          if (lane_idx == 0 and valid_lanes_mask != 0x0)
            subdomain_offsets[warp_idx] = local_warp_offset;
          // Store local offsets (only for valid lanes, otherwise set to -1) :
          //connected_comp_offsets[idx.local] = lane_idx > 0 ? local_warp_offset : 0;
          if (component_size > 0) {
            const auto low_lane_idx = get_ls1b_idx(acc, valid_lanes_mask);
            connected_comp_offsets[idx.local] = lane_idx != low_lane_idx ? local_warp_offset : 0;
          } else {
            connected_comp_offsets[idx.local] = -1;
          }
        }
        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const auto active_lanes_mask = alpaka::warp::ballot(acc, true);
          //
          const auto warp_idx = idx.local / w_extent;
          // Skip inactive warps:
          if (warp_idx != 0)
            continue;
          //
          const auto warp_content_size = subdomain_offsets[idx.local];
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          const auto warp_offset = warp_exclusive_sum(acc, active_lanes_mask, warp_content_size, idx.local);

          subdomain_offsets[idx.local] = warp_offset;  //NOTE: lane 0 get total nnz
        }
        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned warp_offset = lane_idx == 0 and warp_idx > 0 ? subdomain_offsets[warp_idx] : 0;
          const int lane_offset = connected_comp_offsets[idx.local];  // -1 for child vertices
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          // We need to exclude void lanes (all ones that correponds to offset -1):
          const auto valid_lanes_mask = warp::ballot_mask(acc, active_lanes_mask, lane_offset != -1);
          //
          if (lane_offset == -1)
            continue;

          // we need to broadcast the global warp offset (for all warps except warp 0):
          const unsigned shift = warp_idx > 0 ? warp::shfl_mask(acc, valid_lanes_mask, warp_offset, 0, w_extent) : 0;
          //
          const unsigned global_offset = lane_offset + shift;
          // We just need to sync threads in the warp,
          warp::syncWarpThreads_mask(acc, valid_lanes_mask);
          // Store final offsets in shared memory:
          connected_comp_offsets[idx.local] = global_offset;
          // We need this extra step for future cycles:
          const auto low_idx = get_ls1b_idx(acc, valid_lanes_mask);
          if (low_idx != 0)
            connected_comp_offsets[idx.local - low_idx] = global_offset;
          // Last entry for the total number of components:
          // Note: warp_idx = 0 lane_idx = 0 has always valid offset (vertex id 0 is always the root!)
          // so this lane always valid:
          if (warp_idx == 0 && lane_idx == 0)
            connected_comp_offsets[nVertices] = subdomain_offsets[0];
        }
        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          auto is_valid_lane = [](const unsigned int mask, const unsigned int lid) -> bool {
            return ((mask >> lid) & 1);
          };
          // Determin actual warp-level work dimension: it coincides with w_extent for all warps
          // except (potentially!) the last one:
          const unsigned int warp_work_extent = alpaka::popcount(acc, active_lanes_mask);
          //
          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;
          //const auto lane_mask  = (1 << lane_idx);
          //
          // Get local coordinate:
          const int begin = connected_comp_offsets[idx.local];
          //
          const unsigned int valid_offsets_mask = warp::ballot_mask(acc, active_lanes_mask, begin != -1);
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          const auto neigh_lane_idx =
              is_valid_lane(valid_offsets_mask, lane_idx)
                  ? get_high_neighbor_logical_lane_idx(acc, active_lanes_mask, valid_offsets_mask, lane_idx)
                  : w_extent;
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          const unsigned int warp_neigh_begin =
              is_valid_lane(valid_offsets_mask, lane_idx)
                  ? warp::shfl_mask(acc, valid_offsets_mask, begin, neigh_lane_idx, w_extent)
                  : begin;
          const unsigned int end = lane_idx == neigh_lane_idx
                                       ? connected_comp_offsets[idx.local + (warp_work_extent - lane_idx)]
                                       : warp_neigh_begin;
          //
          // We need to exclude all vertices that are globally isolated
          // that is, those which belong to connected components with component size equal to one
          const unsigned component_size = end - begin;
          // Determin a custom mask for such vertices:
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          unsigned int valid_vertex_mask = warp::ballot_mask(acc, active_lanes_mask, (component_size != 1));
          // If the warp does not contain valid vertices for processing at all, then skip to the next warp:
          if (valid_vertex_mask == 0x0)
            continue;
          //
          const unsigned int first_valid_lane_idx = get_ls1b_idx(acc, valid_vertex_mask);
          const unsigned int last_valid_lane_idx = get_ms1b_idx(acc, valid_vertex_mask);
          //
          const int cluster_rhf_size =
              is_valid_lane(valid_vertex_mask, lane_idx) ? component_vertex_rhf_sizes[idx.local] : 0;
          const int cluster_rhf_offset =
              is_valid_lane(valid_vertex_mask, lane_idx) ? component_vertex_rhf_offsets[idx.local] : 0;
          //
          // Initialize valid iterative lane index:
          unsigned int iter_lane_idx = first_valid_lane_idx;
          //
          unsigned int iter_pfrhf_offset, iter_pfrhf_size, iter_leftover_pfrhf_size;
          //
          bool update_params = true;
          // Start iterations untill valid vertex mask will be empty:
          while (iter_lane_idx <= last_valid_lane_idx) {
            if (update_params) {
              warp::syncWarpThreads_mask(acc, active_lanes_mask);
              //
              iter_pfrhf_offset =
                  warp::shfl_mask(acc, active_lanes_mask, cluster_rhf_offset, first_valid_lane_idx, w_extent);
              iter_pfrhf_size =
                  warp::shfl_mask(acc, active_lanes_mask, cluster_rhf_size, first_valid_lane_idx, w_extent);
              iter_leftover_pfrhf_size = 0;
              //
              update_params = false;
            }
            const unsigned int pfrhfrac_idx = lane_idx + iter_pfrhf_offset + iter_leftover_pfrhf_size;

            unsigned int detId = 0;
            unsigned int pfrh_idx = 0;

            if (lane_idx < (iter_pfrhf_size - iter_leftover_pfrhf_size)) {
              pfrh_idx = pfRecHitFracs[pfrhfrac_idx].pfrhIdx();
              //
              detId = pfRecHit[pfrh_idx].detId();
            }

            const unsigned int seedIdx = lane_idx == iter_lane_idx ? component_cluster_seeds[idx.local] : 0;

            warp::syncWarpThreads_mask(acc, active_lanes_mask);

            const auto iter_mask = warp::ballot_mask(acc, active_lanes_mask, lane_idx < iter_pfrhf_size);

            const unsigned int seedIdx_ = warp::shfl_mask(acc, iter_mask, seedIdx, iter_lane_idx, w_extent);

            warp::syncWarpThreads_mask(acc, active_lanes_mask);

            const unsigned int candidate_lane_mask = warp::ballot_mask(acc, iter_mask, seedIdx_ == detId);

            const bool is_done = is_valid_lane(candidate_lane_mask, lane_idx);

            if ((iter_pfrhf_size - iter_leftover_pfrhf_size) < warp_work_extent or is_done) {
              if (is_done and (lane_idx < (iter_pfrhf_size - iter_leftover_pfrhf_size))) {
                component_cluster_energies[iter_lane_idx + warp_idx * w_extent] = pfRecHit[pfrh_idx].energy();
              }
              //
              // Determin the next valid vertex:
              // 1. Erase ls1b in the current iterative mask:
              valid_vertex_mask = erase_ls1b(acc, valid_vertex_mask);
              // 2. Compute lowest index of the new ls1b:
              iter_lane_idx = get_ls1b_idx(acc, valid_vertex_mask);
              //
              update_params = true;
            } else {
              iter_leftover_pfrhf_size += warp_work_extent;
            }
          }  // end while over valid vertices.
          //
        }
        //
        alpaka::syncBlockThreads(acc);
        // Now we need to compute offsets to each connected component:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          const auto warp_idx = idx.local / w_extent;
          //
          if (warp_idx > 0)
            continue;
          //
          const auto full_mask = alpaka::warp::ballot(acc, true);
          //
          auto is_valid_lane = [](const unsigned int mask, const unsigned int lid) -> bool {
            return ((mask >> lid) & 1);
          };
          //
          const auto lane_idx = idx.local % w_extent;
          //
          unsigned int inc = 0;

          for (unsigned int i = 0; i < w_items; i++) {
            const auto j = lane_idx + i * w_extent;
            const auto offset = j < nVertices ? connected_comp_offsets[j] : -1;
            //
            warp::syncWarpThreads_mask(acc, full_mask);
            const auto valid_offset_mask = warp::ballot_mask(acc, full_mask, offset != -1);

            if (valid_offset_mask == 0x0)
              continue;
            //
            if (is_valid_lane(valid_offset_mask, lane_idx)) {
              const auto logical_lane_idx = get_logical_lane_idx(acc, valid_offset_mask, lane_idx);
              //
              topol_comp_offsets[logical_lane_idx + inc] = offset;
            }
            inc += alpaka::popcount(acc, valid_offset_mask);
          }
          if (lane_idx == 0)
            topol_comp_offsets[nVertices] = connected_comp_offsets[nVertices];
        }

        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          // Reset shared memory buffers to zero:
          connected_comp_vertices[idx.local] = 0;
          connected_comp_pos[idx.local] = 0;
          //
        }
        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          auto is_root_lane = [](const unsigned int mask, const unsigned int lid) -> bool {
            return ((mask >> lid) & 1);
          };
          //
          // Determin actual warp-level work dimension: it coincides with w_extent for all warps
          // except (potentially!) the last one:
          const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);
          //
          const auto vertex_idx = idx.local;
          //
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned int component_root_idx = component_roots[vertex_idx];
          //
          const bool is_component_root = component_root_idx == vertex_idx;
          //
          warp::syncWarpThreads_mask(acc, active_lanes_mask);
          //
          const auto component_root_mask = warp::ballot_mask(acc, active_lanes_mask, is_component_root);

          if (is_root_lane(component_root_mask, lane_idx) == false)
            continue;

          const unsigned begin = connected_comp_offsets[vertex_idx];
          //
          const unsigned int intern_connected_comp_mask = intern_connected_comp_masks[vertex_idx];
          //
          const auto root_lanes_mask = active_lanes_mask & component_root_mask;
          //
          unsigned int connected_vertex_pos = begin;
          //
          connected_comp_vertices[connected_vertex_pos++] = component_root_idx;

          for (unsigned lid = 0; lid < warp_work_extent; ++lid) {
            const auto target_lane_idx = (intern_connected_comp_mask >> lid) & 1;
            if (target_lane_idx != 0) {
              const unsigned int connected_vertex_idx = lid + warp_idx * w_extent;
              connected_comp_vertices[connected_vertex_pos++] = connected_vertex_idx;
            }
          }
          //
          warp::syncWarpThreads_mask(acc, root_lanes_mask);
          //
          connected_comp_pos[component_root_idx] = connected_vertex_pos;
        }

        //
        alpaka::syncBlockThreads(acc);
        //
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          //
          unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;
          //
          // Determin actual warp-level work dimension: it coincides with w_extent for all warps
          // except (potentially!) the last one:
          const auto warp_work_extent = warp::ballot_mask(acc, active_lanes_mask, true);
          //
          const auto vertex_idx = idx.local;

          const auto warp_idx = idx.local / w_extent;
          // Get local coordinate:
          const unsigned int extern_connected_comp_mask = extern_connected_comp_masks[idx.local];

          const auto nnz = alpaka::popcount(acc, extern_connected_comp_mask);

          warp::syncWarpThreads_mask(acc, active_lanes_mask);

          if (nnz == 0)
            continue;  // skip inactive lanes
          //
          const unsigned int component_root_idx = component_roots[vertex_idx];

          unsigned int connected_vertex_pos =
              alpaka::atomicAdd(acc, &connected_comp_pos[component_root_idx], nnz, alpaka::hierarchy::Threads{});

          for (unsigned lid = 0; lid < warp_work_extent; ++lid) {
            const auto target_lane_idx =
                (extern_connected_comp_mask >> lid) & 1;  // number of target lanes is equal to nnz
            if (target_lane_idx != 0) {
              const unsigned int connected_vertex_idx = lid + warp_idx * w_extent;
              connected_comp_vertices[connected_vertex_pos++] = connected_vertex_idx;
            }
          }
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const auto warp_idx = idx.local / w_extent;
          const auto lane_idx = idx.local % w_extent;
          //
          const unsigned int nTopos = topol_comp_offsets[nVertices];
          //
          unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nTopos);
          //
          const auto c = idx.local;
          //
          if (idx.local >= nTopos)
            continue;
          //
          const unsigned int begin = topol_comp_offsets[idx.local];
          const unsigned int end = idx.local < (nTopos - 1) ? topol_comp_offsets[idx.local + 1] : nTopos;
          const unsigned int component_size = end - begin;
          //
          const unsigned int root_idx = connected_comp_vertices[begin];
          //
          outPFCluster[c].depth() = pfCluster[root_idx].depth();
          outPFCluster[c].topoId() = pfCluster[root_idx].topoId();
          outPFCluster[c].energy() = pfCluster[root_idx].energy();
          outPFCluster[c].x() = pfCluster[root_idx].x();
          outPFCluster[c].y() = pfCluster[root_idx].y();
          outPFCluster[c].z() = pfCluster[root_idx].z();
          outPFCluster[c].topoRHCount() = pfCluster[root_idx].topoRHCount();
          //
          auto rhfracSize = pfCluster[root_idx].rhfracSize();
          //
          float root_energy = component_cluster_energies[root_idx];
          //
          unsigned int seed_component_idx = root_idx;
          //
          for (unsigned int i = 1; i < component_size; i++) {
            const unsigned int component_idx = connected_comp_vertices[root_idx + i];
            rhfracSize += pfCluster[component_idx].rhfracSize();
            const float component_energy = component_cluster_energies[component_idx];
            if (component_energy > root_energy) {
              root_energy = component_energy;
              seed_component_idx = component_idx;
            }
          }
          //
          outPFCluster[c].rhfracSize() = rhfracSize;
          //
          if (seed_component_idx != root_idx)
            outPFCluster[c].seedRHIdx() = pfCluster[seed_component_idx].seedRHIdx();
          //
          const auto local_warp_offset = warp_exclusive_sum(acc, active_lanes_mask, rhfracSize, lane_idx);
          // Store warp offsets in a separate buffer:
          if (lane_idx == 0)
            subdomain_offsets[warp_idx] = local_warp_offset;
          // Second, store total local offsets into shared mem buffer:
          connected_comp_offsets[idx.local] = lane_idx > 0 ? local_warp_offset : 0;
        }
        //
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
          const auto local_warp_stride = lane_idx < w_items ? subdomain_offsets[lane_idx] : 0;
          //
          const auto global_warp_offset = warp_exclusive_sum(acc, full_mask, local_warp_stride, lane_idx);

          if (lane_idx < w_items)
            subdomain_offsets[lane_idx] = global_warp_offset;  //NOTE: lane 0 get total nnz
          //
          if (lane_idx == 0) {
            const auto nTopos = topol_comp_offsets[nVertices];
            connected_comp_offsets[nTopos] = global_warp_offset;
          }
        }
        alpaka::syncBlockThreads(acc);
        // Now we assemble global offsets for each vertex:
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const unsigned int nTopos = topol_comp_offsets[nVertices];
          //
          if (idx.local >= nTopos)
            continue;
          //
          const unsigned int begin_c = topol_comp_offsets[idx.local];
          const unsigned int end_c = idx.local < (nTopos - 1) ? topol_comp_offsets[idx.local + 1] : nTopos;
          //
          unsigned int iter_pfrh_idx = connected_comp_offsets[idx.local];
          //
          for (unsigned int j = begin_c; j < end_c; j++) {
            const unsigned int pfc_idx = connected_comp_vertices[j];
            //
            const unsigned int begin_rhfrac = pfCluster[pfc_idx].rhfracOffset();
            const unsigned int end_rhfrac = pfCluster[pfc_idx].rhfracSize();
            //
            for (unsigned int l = begin_rhfrac; l < end_rhfrac; l++) {
              outPFRecHitFracs[iter_pfrh_idx].frac() = pfRecHitFracs[l].frac();
              outPFRecHitFracs[iter_pfrh_idx].pfrhIdx() = pfRecHitFracs[l].pfrhIdx();
              outPFRecHitFracs[iter_pfrh_idx].pfcIdx() = pfRecHitFracs[l].pfcIdx();
              iter_pfrh_idx += 1;
            }
          }
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
