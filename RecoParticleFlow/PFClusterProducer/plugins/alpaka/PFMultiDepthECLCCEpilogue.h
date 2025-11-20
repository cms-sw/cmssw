#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogue_h

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

  using namespace ::cms::alpakatools;

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
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFClusterDeviceCollection::View outPFCluster,
        reco::PFRecHitFractionDeviceCollection::View outPFRecHitFracs,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels,
        const reco::PFClusterDeviceCollection::ConstView pfCluster,
        const reco::PFRecHitFractionDeviceCollection::ConstView pfRecHitFracs,
        const reco::PFRecHitDeviceCollection::ConstView pfRecHit) const {
      static_assert(max_w_items <= 32,
                    "ECLCCEpilogueKernel: Maximum number of supported warps per block is 32, "
                    "assuming one warp per 32 threads.");
      constexpr unsigned int max_w_extent = 32;  // max warp size..

      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int w_extent = alpaka::warp::getSize(acc);

      auto& component_roots(alpaka::declareSharedVar<int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));

      auto& connected_comp_rhf_offsets(alpaka::declareSharedVar<int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));

      auto& cc_roots(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));
      auto& cc_energies(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent], __COUNTER__>(acc));
      auto& cc_seeds(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent], __COUNTER__>(acc));
      auto& cc_rhf_sizes(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));

      auto& component_map(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));

      auto& vertex_seeds(alpaka::declareSharedVar<int[max_w_items * max_w_extent], __COUNTER__>(acc));

      auto& tmp_buf1(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));
      auto& tmp_buf2(alpaka::declareSharedVar<unsigned int[max_w_items * max_w_extent + 1], __COUNTER__>(acc));
      auto& tmp_buf3(alpaka::declareSharedVar<unsigned int[max_w_items], __COUNTER__>(acc));

      auto& connected_comp_sizes = tmp_buf1;
      auto& connected_comp_rhf_sizes = tmp_buf2;
      auto& subcc_offsets = tmp_buf3;

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        if (::cms::alpakatools::once_per_block(acc)) {
	  component_map[nVertices] = 0;	
        }

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          vertex_seeds[idx.local] = 0;

          connected_comp_sizes[idx.local] = 0;
          connected_comp_rhf_sizes[idx.local] = 0;

          if (idx.local < max_w_items)
            subcc_offsets[idx.local] = 0;
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          if (idx.local >= nVertices)
            continue;

          const auto vertex_idx = idx.local;

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int rep_idx = pfClusteringCCLabels[vertex_idx].mdpf_topoId();

          component_roots[vertex_idx] = rep_idx;

          const bool is_representative = vertex_idx == rep_idx;

          const warp::warp_mask_t rep_mask = warp::ballot_mask(acc, active_lanes_mask, is_representative);

          if (is_work_lane(rep_mask, lane_idx, w_extent) == false)
            continue;

          const unsigned int low_rep_idx = get_ls1b_idx(acc, rep_mask);

          unsigned int local_topo_offset = 0;

          if (lane_idx == low_rep_idx) {
            const unsigned int local_n_topo = alpaka::popcount(acc, rep_mask);

            local_topo_offset =
                alpaka::atomicAdd(acc, &component_map[nVertices], local_n_topo, alpaka::hierarchy::Threads{});
          }
          warp::syncWarpThreads_mask(acc, rep_mask);

          const unsigned int cc_idx_stub = warp::shfl_mask(acc, rep_mask, local_topo_offset, low_rep_idx, w_extent);
          const unsigned int cc_idx = cc_idx_stub + get_logical_lane_idx(acc, rep_mask, lane_idx);

          cc_roots[cc_idx] = rep_idx;
          component_map[vertex_idx] = cc_idx;
        }
        alpaka::syncBlockThreads(acc);

        const unsigned int nComponents = component_map[nVertices];

        // Store nComponents in the global buffer (per block):
        if (::cms::alpakatools::once_per_block(acc)) {
          outPFCluster.nTopos() = nComponents;
          outPFCluster.nSeeds() = nComponents;
          outPFCluster.size() = nComponents;
        }

        // Store relevant data (per thread):
        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nComponents, w_extent))) {
          if (idx.local >= nComponents)
            continue;

          const unsigned int topo_idx = idx.local;
          const unsigned int root_idx = cc_roots[topo_idx];

          outPFCluster[topo_idx].depth() = pfCluster[root_idx].depth();
          outPFCluster[topo_idx].topoId() = topo_idx;
          outPFCluster[topo_idx].energy() = pfCluster[root_idx].energy();
          outPFCluster[topo_idx].x() = pfCluster[root_idx].x();
          outPFCluster[topo_idx].y() = pfCluster[root_idx].y();
          outPFCluster[topo_idx].z() = pfCluster[root_idx].z();
          outPFCluster[topo_idx].topoRHCount() = pfCluster[root_idx].topoRHCount();  //??
        }

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          // Skip inactive lanes:
          if (idx.local >= nVertices)
            continue;

          const auto vertex_idx = idx.local;

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int rep_idx = component_roots[vertex_idx];

          const warp::warp_mask_t subcomp_mask = warp::match_any_mask(acc, active_lanes_mask, rep_idx);
          const unsigned int local_subcomponent_rep_idx = get_ls1b_idx(acc, subcomp_mask);

          if (lane_idx == local_subcomponent_rep_idx) {
            const unsigned int subcomp_size = alpaka::popcount(acc, subcomp_mask);

            alpaka::atomicAdd(acc, &connected_comp_sizes[rep_idx], subcomp_size, alpaka::hierarchy::Threads{});
          }
          // Load rhfrac sizes (to compute offsets for rechit fractions):
          const unsigned int rhf_size = pfCluster[vertex_idx].rhfracSize();

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

          connected_comp_rhf_offsets[vertex_idx] = relative_rhf_offset + subcomp_rhf_offset;  // store relative offsets
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);

          if (idx.local >= nVertices)
            continue;

          if (idx.local == 0) {
            cc_roots[nVertices] = nComponents;  // hold its size
          }

          const auto vertex_idx = idx.local;

          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int rep_idx = component_roots[vertex_idx];
          const bool is_representative = vertex_idx == rep_idx;

          const warp::warp_mask_t rep_mask = warp::ballot_mask(acc, active_lanes_mask, is_representative);
          vertex_seeds[vertex_idx] = pfCluster[idx.local].seedRHIdx();

          const unsigned int connected_comp_rhf_size = is_representative ? connected_comp_rhf_sizes[rep_idx] : 0;

          if (is_work_lane(rep_mask, lane_idx, w_extent)) {
            const unsigned int cc_idx = component_map[vertex_idx];
            cc_rhf_sizes[cc_idx] = connected_comp_rhf_size;
          }
        }

        alpaka::syncBlockThreads(acc);

        auto& cc_rhf_offsets = tmp_buf2;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nComponents, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nComponents);

          if (idx.local >= nComponents)
            continue;

          const unsigned int warp_idx = idx.local / w_extent;
          const unsigned int lane_idx = idx.local % w_extent;

          const unsigned int cc_rhf_size = cc_rhf_sizes[idx.local];  //topo_id = idx.local

          const unsigned int cc_rhf_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, cc_rhf_size, lane_idx);
          cc_rhf_offsets[idx.local] = lane_idx == 0 ? 0 : cc_rhf_offset;

          if (lane_idx == 0)
            subcc_offsets[warp_idx] = cc_rhf_offset;
        }
        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(w_extent, w_extent))) {
          const unsigned int cc_rhf_size = subcc_offsets[idx.local];

          const unsigned int cc_rhf_global_offset = warp_exclusive_sum(acc, cc_rhf_size, idx.local);

          subcc_offsets[idx.local] = cc_rhf_global_offset;
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nComponents, w_extent))) {
          if (idx.local >= nComponents)
            continue;

          const unsigned int warp_idx = idx.local / w_extent;

          const unsigned int cc_rhf_global_offset = warp_idx == 0 ? 0 : subcc_offsets[warp_idx];

          if (warp_idx > 0) {
            cc_rhf_offsets[idx.local] += cc_rhf_global_offset;
          }
        }
        alpaka::syncBlockThreads(acc);

        auto& vertex_energies = tmp_buf1;  //WARNING: we treat this buffer as connected_comp_sizes (read) to get
                                           //         component sizes, which is needed to isolate trivial components,
                                           //         and then as vertex_energies (write) to store rh energies;
                                           //         all this operations are thread local, so are harmless.
        auto& isolated_roots = tmp_buf3;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);

          if (idx.local >= nVertices)
            continue;

          const unsigned int lane_idx = idx.local % w_extent;
          const unsigned int warp_idx = idx.local / w_extent;

          const unsigned int vertex_idx = idx.local;

          const unsigned int rep_idx = component_roots[vertex_idx];
          const unsigned int cc_idx = component_map[rep_idx];

          const unsigned int rhf_begin = pfCluster[idx.local].rhfracOffset();
          const unsigned int rhf_end = rhf_begin + pfCluster[vertex_idx].rhfracSize();

          const bool is_root = (vertex_idx == rep_idx);

          const unsigned int comp_size = is_root ? connected_comp_sizes[vertex_idx] : 0;

          const bool is_isolated_root = is_root && (comp_size == 1);

          const warp::warp_mask_t iso_root_mask = warp::ballot_mask(acc, active_lanes_mask, is_isolated_root);

          const unsigned int seed = vertex_seeds[vertex_idx];

          if (lane_idx == 0)
            isolated_roots[warp_idx] = iso_root_mask;

          unsigned int store_idx = connected_comp_rhf_offsets[vertex_idx] + cc_rhf_offsets[cc_idx];

          float energy = is_isolated_root == false ? pfRecHit[seed].energy() : 0.f;

          for (unsigned int j = rhf_begin; j < rhf_end; j++) {
            outPFRecHitFracs[store_idx].frac() = pfRecHitFracs[j].frac();
            outPFRecHitFracs[store_idx].pfrhIdx() = pfRecHitFracs[j].pfrhIdx();
            outPFRecHitFracs[store_idx].pfcIdx() = cc_idx;
            ++store_idx;
          }

          auto compFn = [] ALPAKA_FN_ACC(const float a, const float b) -> float { return a > b ? a : b; };

          const warp::warp_mask_t updated_active_lanes_mask =
              warp::ballot_mask(acc, active_lanes_mask, is_isolated_root == false);

          if (is_isolated_root)
            continue;

          const warp::warp_mask_t subcomponent_mask = warp::match_any_mask(acc, updated_active_lanes_mask, rep_idx);

          const unsigned int low_local_rep_idx = get_ls1b_idx(acc, subcomponent_mask);

          const float max_energy = warp_sparse_reduce(acc, subcomponent_mask, lane_idx, energy, compFn);

          const warp::warp_mask_t max_energy_lanes_mask =
              warp::ballot_mask(acc, subcomponent_mask, max_energy == energy);

          const unsigned int max_energy_lane_idx = alpaka::ffs(acc, static_cast<int>(max_energy_lanes_mask)) - 1;

          //need to store seed:
          const auto seed_max = warp::shfl_mask(acc, subcomponent_mask, seed, max_energy_lane_idx, w_extent);
          const auto energy_max = warp::shfl_mask(acc, subcomponent_mask, max_energy, max_energy_lane_idx, w_extent);

          warp::syncWarpThreads_mask(acc, updated_active_lanes_mask);

          if (lane_idx == low_local_rep_idx) {
            unsigned int x = std::bit_cast<unsigned int>(energy_max);
            alpaka::atomicMax(acc, &cc_energies[cc_idx], x);
            vertex_seeds[vertex_idx] = seed_max;
            vertex_energies[vertex_idx] = x;
          }
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nVertices, w_extent))) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx.local < nVertices);
          if (idx.local >= nVertices)
            continue;

          const unsigned int lane_idx = idx.local % w_extent;
          const unsigned int warp_idx = idx.local / w_extent;

          const warp::warp_mask_t iso_root_mask = isolated_roots[warp_idx];

          const unsigned int vertex_idx = idx.local;

          const unsigned int rep_idx = component_roots[vertex_idx];
          const unsigned int cc_idx = component_map[rep_idx];

          const bool is_isolated_root = is_work_lane(iso_root_mask, lane_idx, w_extent);

          const warp::warp_mask_t updated_active_lanes_mask =
              warp::ballot_mask(acc, active_lanes_mask, is_isolated_root == false);

          if (is_isolated_root) {
            cc_seeds[cc_idx] = vertex_seeds[vertex_idx];
            continue;
          }

          const warp::warp_mask_t subcomponent_mask = warp::match_any_mask(acc, updated_active_lanes_mask, rep_idx);
          const unsigned int low_local_rep_idx = get_ls1b_idx(acc, subcomponent_mask);

          if (lane_idx == low_local_rep_idx) {
            const unsigned int probe_energy = vertex_energies[vertex_idx];
            const unsigned int max_energy = cc_energies[cc_idx];
            // We have only one vertex that holds max energy, so no race condition here:
            if (probe_energy == max_energy)
              cc_seeds[cc_idx] = vertex_seeds[vertex_idx];
          }
        }
        alpaka::syncBlockThreads(acc);

        for (auto idx : ::cms::alpakatools::uniform_group_elements(
                 acc, group, ::cms::alpakatools::round_up_by(nComponents, w_extent))) {
          if (idx.local >= nComponents)
            continue;

          const unsigned int topo_idx = idx.local;

          outPFCluster[topo_idx].rhfracOffset() = cc_rhf_offsets[topo_idx];
          outPFCluster[topo_idx].rhfracSize() = cc_rhf_sizes[topo_idx];
          outPFCluster[topo_idx].seedRHIdx() = cc_seeds[topo_idx];
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
