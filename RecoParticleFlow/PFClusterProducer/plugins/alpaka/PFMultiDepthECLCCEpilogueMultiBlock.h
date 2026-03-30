#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogueMultiBlock_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthECLCCEpilogueMultiBlock_h

#include <limits>

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

  class ECLCCEpilogueRecHitFracOffsetsKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View args,
        const reco::PFMultiDepthClusteringCCLabelsDeviceCollection::ConstView pfClusteringCCLabels,
        const reco::PFClusterDeviceCollection::ConstView pfCluster) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      const unsigned int nVertices = pfClusteringCCLabels.size();

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

          const unsigned int vertex_idx = idx.global;
          const unsigned int rep_idx = pfClusteringCCLabels[vertex_idx].mdpf_topoId();
          const warp::warp_mask_t rep_lane_mask = get_lane_mask(rep_idx % w_extent);

          const bool is_representative = vertex_idx == rep_idx;

          // Load rhfrac sizes (to compute offsets for rechit fractions):
          const unsigned int rhf_size = pfCluster[vertex_idx].rhfracSize();

          const unsigned int lane_idx = idx.local % w_extent;

          const warp::warp_mask_t subcomp_mask = warp::match_any_mask(acc, active_lanes_mask, rep_idx);

          const unsigned int subcomp_rep_idx = get_ls1b_idx(acc, subcomp_mask);

          bool update_iso_root_lane = false;

          unsigned int which_global_warp_idx = std::numeric_limits<unsigned int>::max();

          if (lane_idx == subcomp_rep_idx) {
            const unsigned int subcomp_size = alpaka::popcount(acc, subcomp_mask);
            if ((subcomp_size > 1) || (subcomp_size == 1 && !is_representative)) {
              update_iso_root_lane = true;
              which_global_warp_idx = rep_idx / w_extent;
            }
          }

          const warp::warp_mask_t iso_root_lanes = warp::ballot_mask(acc, active_lanes_mask, update_iso_root_lane);

          const warp::warp_mask_t iso_root_lanes_subgroup =
              warp::match_any_mask(acc, active_lanes_mask, which_global_warp_idx);

          if (is_work_lane(iso_root_lanes, lane_idx)) {
            // Construct correct rep mask:
            auto orFn = [] ALPAKA_FN_ACC(const warp::warp_mask_t m1, const warp::warp_mask_t m2) -> warp::warp_mask_t {
              return m1 | m2;
            };

            warp::warp_mask_t selected_iso_root_mask =
                warp_sparse_reduce(acc, iso_root_lanes_subgroup, lane_idx, rep_lane_mask, orFn);

            if (is_ls1b_idx<Acc1D>(iso_root_lanes_subgroup, lane_idx)) {
              // Temporary WAR (using De Morgan's law):
              const warp::warp_mask_t non_isolated_vertex_lanes = ~selected_iso_root_mask;

              alpaka::atomicAnd(acc,
                                &args[which_global_warp_idx].vertexMask(),  //must be init to full mask!
                                non_isolated_vertex_lanes,
                                alpaka::hierarchy::Threads{});
            }
          }

          unsigned int subcomp_rhf_offset = warp_sparse_exclusive_sum(acc, subcomp_mask, rhf_size, lane_idx);

          unsigned int relative_rhf_offset_stub = 0;

          if (lane_idx == subcomp_rep_idx) {
            // Remark: exclusive sum returns total number of elements
            // (i.e., subcomponent rhf size) for the lowest lane idx in the mask.
            relative_rhf_offset_stub =
                alpaka::atomicAdd(acc, &args[rep_idx].ccRHFSize(), subcomp_rhf_offset, alpaka::hierarchy::Threads{});
            subcomp_rhf_offset = 0;  // we need to reset local offset for local rep lane.
          }

          const unsigned int relative_rhf_offset =
              warp::shfl_mask(acc, subcomp_mask, relative_rhf_offset_stub, subcomp_rep_idx, w_extent);
          //connected comp rhf offsets for all vertices:
          args[vertex_idx].ccRHFOffset() = relative_rhf_offset + subcomp_rhf_offset;  // store relative offsets
        }
      }
    }
  };

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

  template <unsigned int max_w_items = 32>
  class ECLCCEpilogueCCOffsetsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFClusterDeviceCollection::View outPFCluster,
                                  reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View args,
                                  reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
                                  const reco::PFClusterDeviceCollection::ConstView pfCluster) const {
      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      static_assert(max_w_items <= 32, "ECLCCEpilogueKernel: number of warps per block is unsupported.");

      const unsigned int nVertices = pfClusteringCCLabels.size();

      const unsigned int blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

      const unsigned int gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];
      const unsigned int nBlocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];

      // component offsets
      auto& subcc_offsets(alpaka::declareSharedVar<unsigned int[max_w_items], __COUNTER__>(acc));

      auto& common_buf1(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));
      auto& common_buf2(alpaka::declareSharedVar<unsigned int[max_w_items * w_extent], __COUNTER__>(acc));

      //block-local number of components
      unsigned int& localNComponents = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      unsigned int& isLastBlockDone = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      unsigned int& topo_block_offset = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        if (::cms::alpakatools::once_per_block(acc)) {
          localNComponents = 0;
          isLastBlockDone = 0;
          topo_block_offset = 0;
        }

        unsigned int vertex_idx = nVertices;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          if (idx.local < max_w_items) {
            subcc_offsets[idx.local] = 0;
          }
          vertex_idx = idx.global;
        }

        const unsigned int rep_idx = vertex_idx < nVertices ? pfClusteringCCLabels[vertex_idx].mdpf_topoId() : gridDim;

        const bool is_representative = vertex_idx < nVertices ? vertex_idx == rep_idx : false;

        auto& block_local_component_map = common_buf1;
        auto& block_local_cc_idx_record = common_buf2;

        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
          const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);
          const unsigned int lane_idx = idx.local % w_extent;

          const warp::warp_mask_t rep_mask = warp::ballot_mask(acc, active_lanes_mask, is_representative);

          if (is_representative == false) {
            block_local_component_map[idx.local] = nVertices;
            block_local_cc_idx_record[idx.local] = 0;
            continue;
          }

          const unsigned int low_rep_idx = get_ls1b_idx(acc, rep_mask);

          unsigned int local_topo_offset = 0;

          if (lane_idx == low_rep_idx) {
            const unsigned int local_n_topo = alpaka::popcount(acc, rep_mask);

            local_topo_offset = alpaka::atomicAdd(acc, &localNComponents, local_n_topo, alpaka::hierarchy::Threads{});
          }

          const unsigned int cc_idx_stub = warp::shfl_mask(acc, rep_mask, local_topo_offset, low_rep_idx, w_extent);
          const unsigned int cc_idx = cc_idx_stub + get_logical_lane_idx(acc, rep_mask, lane_idx);

          block_local_cc_idx_record[cc_idx] = rep_idx;
          block_local_component_map[idx.local] = cc_idx;
        }
        alpaka::syncBlockThreads(acc);

        const unsigned int nBlockLocalComponents = localNComponents;

        if (::cms::alpakatools::once_per_block(acc)) {
          topo_block_offset = alpaka::atomicAdd(acc,
                                                &pfClusteringCCLabels.ncomponents(),
                                                static_cast<int>(nBlockLocalComponents),
                                                alpaka::hierarchy::Threads{});

          alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
        }

        alpaka::syncBlockThreads(acc);
        // block/group uniform branch (mandatary for all threads in the block):
        if (nBlockLocalComponents != 0) {
          unsigned int global_topo_idx = nVertices;

          // Store relevant data (per thread):
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            if (idx.local >= nBlockLocalComponents)
              continue;
            const unsigned int root_idx = block_local_cc_idx_record[idx.local];  //collection of rep indices
            global_topo_idx = idx.local + topo_block_offset;

            outPFCluster[global_topo_idx].depth() = pfCluster[root_idx].depth();
            outPFCluster[global_topo_idx].topoId() = global_topo_idx;
            outPFCluster[global_topo_idx].energy() = pfCluster[root_idx].energy();
            outPFCluster[global_topo_idx].x() = pfCluster[root_idx].x();
            outPFCluster[global_topo_idx].y() = pfCluster[root_idx].y();
            outPFCluster[global_topo_idx].z() = pfCluster[root_idx].z();
            outPFCluster[global_topo_idx].topoRHCount() = pfCluster[root_idx].topoRHCount();
            // note that root idx is within the block range:
            args[root_idx].rootMap() = global_topo_idx;
            // now we can reset common_buf2 buffer (block_local_cc_idx_record):
            common_buf2[idx.local] = 0;
          }

          alpaka::syncBlockThreads(acc);

          // total rhfrac count per component (compact indexing).
          auto& cc_rhf_sizes = common_buf2;

          if (is_representative) {  // rep_cc_idx in the range [0,nBlockLocalComponents)
            const unsigned int rep_cc_idx = block_local_component_map[rep_idx % blockDim];
            args[rep_idx].rootLocalMap() = rep_cc_idx;
            cc_rhf_sizes[rep_cc_idx] = args[rep_idx].ccRHFSize();
          }

          alpaka::syncBlockThreads(acc);

          unsigned int cc_rhf_offset = 0;

          // Compute block-local rhf offsets:
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            if (idx.local >= nBlockLocalComponents)
              continue;
            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;

            const unsigned int cc_rhf_size = cc_rhf_sizes[idx.local];
            // Store rhf sizes in memory:
            outPFCluster[global_topo_idx].rhfracSize() = cc_rhf_size;

            cc_rhf_offset = warp_sparse_exclusive_sum(
                acc, active_lanes_mask, cc_rhf_size, lane_idx);  //warp local rhf offsets, lane zero contains NNZ

            if (lane_idx == 0) {
              subcc_offsets[warp_idx] = cc_rhf_offset;  //store warp-local nnz
              cc_rhf_offset = 0;
            }
          }
          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, w_extent)) {
            if (nBlockLocalComponents == 1 || idx.local > max_w_items)
              continue;

            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int cc_warp_nnz = subcc_offsets[idx.local];  //load nnz per warp

            const unsigned int cc_rhf_global_offset =
                warp_sparse_exclusive_sum(acc, active_lanes_mask, cc_warp_nnz, idx.local);

            subcc_offsets[idx.local] = cc_rhf_global_offset;  //subcc_offsets[0] contains block nnz
          }

          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            if (idx.local >= nBlockLocalComponents)
              continue;
            const unsigned int warp_idx = idx.local / w_extent;

            const unsigned int cc_rhf_block_offset = warp_idx > 0 || idx.local == 0 ? subcc_offsets[warp_idx] : 0;

            cc_rhf_offset += cc_rhf_block_offset;

            args[idx.global].blockRHFOffset() = cc_rhf_offset;  //idx.local = 0 has block nnz
          }
        }  // end of nBlockLocalComponents branch

        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

        if (::cms::alpakatools::once_per_block(acc)) {
          const unsigned int whichBlockDone =
              alpaka::atomicAdd(acc, &args.blockCount(), +1, alpaka::hierarchy::Threads{});

          alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

          if (whichBlockDone == (nBlocks - 1)) {
            isLastBlockDone = 1;
          }
        }

        alpaka::syncBlockThreads(acc);

        if (isLastBlockDone) {
          if (::cms::alpakatools::once_per_block(acc)) {
            const unsigned int totNComponents = pfClusteringCCLabels.ncomponents();

            outPFCluster.nTopos() = totNComponents;
            outPFCluster.nSeeds() = totNComponents;
            outPFCluster.nRHFracs() = pfCluster.nRHFracs();
            outPFCluster.size() = totNComponents;
          }

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nVertices)) {
            common_buf2[idx.local] = 0;
          }
          alpaka::syncBlockThreads(acc);

          auto& reduce_buf = common_buf2;
          //warning nBlocks must be <= blockDim
          unsigned int block_offset = 0;

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, gridDim)) {
            if (idx.local >= nBlocks)
              continue;

            const warp::warp_mask_t active_lanes_mask = alpaka::warp::activemask(acc);

            const unsigned int warp_idx = idx.local / w_extent;
            const unsigned int lane_idx = idx.local % w_extent;

            const unsigned int load_idx = idx.local * blockDim;

            const unsigned int block_rhf_size = args[load_idx].blockRHFOffset();

            block_offset = warp_sparse_exclusive_sum(acc, active_lanes_mask, block_rhf_size, lane_idx);

            if (lane_idx == 0) {
              reduce_buf[warp_idx] = block_offset;  //store warp-local nnz
              block_offset = 0;
            }
          }

          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, w_extent)) {
            if (nBlocks < w_extent)
              continue;

            const unsigned int tmp = reduce_buf[idx.local];  //load nnz per warp

            const unsigned int global_offset = warp_exclusive_sum(acc, tmp, idx.local);

            reduce_buf[idx.local] = global_offset;  //reduce_buf[0] contains total nnz
          }

          alpaka::syncBlockThreads(acc);

          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, gridDim)) {
            if (idx.local >= nBlocks)
              continue;
            const unsigned int warp_idx = idx.local / w_extent;

            const unsigned int shift = warp_idx > 0 ? reduce_buf[warp_idx] : 0;

            block_offset += shift;

            // now re-use buffer for the offsets:
            const unsigned int load_idx = idx.local * blockDim;

            args[load_idx].blockRHFOffset() = block_offset;
          }
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
