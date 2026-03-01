#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterECLCC_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterECLCC_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterWarpIntrinsics.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"

/*
  ECL-CC code: ECL-CC is a connected components graph algorithm. The CUDA
  implementation thereof is quite fast. It operates on graphs stored in
  binary CSR format.

  Copyright (c) 2017-2020, Texas State University. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Texas State University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  Authors: Jayadharini Jaiganesh and Martin Burtscher

  URL: The latest version of this code is available at
  https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/.

  Publication: This work is described in detail in the following paper.
  Jayadharini Jaiganesh and Martin Burtscher. A High-Performance Connected
  Components Implementation for GPUs. Proceedings of the 2018 ACM International
  Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
  June 2018.
*/

/**
 * @file PFMultiDepthClusterECLCC.h
 * @brief Efficient Connected Components Labeling (ECL-CC) for multi-depth particle flow clustering.
 * 
 * This header defines the Alpaka ECL-CC algorithm adapted for particle flow clustering graphs.
 * It partitions the work dynamically based on vertex degree and applies warp-local, warp-wide, and
 * block-wide hooking strategies to efficiently find connected components.
 *
 * The kernel stages include:
 * - Initialization by linking each vertex to its minimum neighbor.
 * - Low-degree vertex processing directly at warp scope.
 * - Dynamic worklist creation for mid-degree and high-degree vertices.
 * - Warp-wide processing for mid-degree vertices.
 * - Block-wide processing for high-degree vertices.
 * - Final flattening phase (pointer jumping) to ensure correct labeling.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc {

  using namespace ::cms::alpakatools;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void hook(
      Acc1D const& acc,
      reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
      const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars,
      const int v,
      const int begin_v,
      const int end_v,
      const int offset_v) {
    auto representative = [pfClusteringCCLabelsPtr = &pfClusteringCCLabels](const int v) -> int {
      int curr_v = (*pfClusteringCCLabelsPtr)[v].mdpf_topoId();

      if (curr_v == v)
        return curr_v;

      int prev_v = v;
      int next_v = (*pfClusteringCCLabelsPtr)[curr_v].mdpf_topoId();

      while (curr_v > next_v) {
        (*pfClusteringCCLabelsPtr)[prev_v].mdpf_topoId() = next_v;
        prev_v = curr_v;
        curr_v = next_v;
        next_v = (*pfClusteringCCLabelsPtr)[curr_v].mdpf_topoId();
      }

      return curr_v;
    };

    int rep_v = representative(v);

    for (int w = begin_v; w < end_v; w += offset_v) {
      const int neigh_v = pfClusteringEdgeVars[w].mdpf_adjacencyList();

      if (v <= neigh_v)
        continue;

      int rep_neigh_v = representative(neigh_v);

      while (rep_v != rep_neigh_v) {
        const bool is_low = (rep_v < rep_neigh_v);

        int low_rep = is_low ? rep_v : rep_neigh_v;
        int high_rep = !is_low ? rep_v : rep_neigh_v;

        int tmp = alpaka::atomicCas(
            acc, &pfClusteringCCLabels[high_rep].mdpf_topoId(), high_rep, low_rep, alpaka::hierarchy::Blocks{});

        if (tmp == high_rep)
          break;  // merge successful, exit.

        if (is_low)
          rep_neigh_v = tmp;
        else
          rep_v = tmp;
      }
    }
  }

  // ECL-CC algorithm driver:
  class ECLCCInitKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars) const {
      const unsigned int nClusters = pfClusteringCCLabels.size();

      if (::cms::alpakatools::once_per_grid(acc)) {
        pfClusteringCCLabels.topH() = nClusters - 1;
        pfClusteringCCLabels.posH() = nClusters - 1;
        pfClusteringCCLabels.topL() = 0;
        pfClusteringCCLabels.posL() = 0;
      }

      for (int v : ::cms::alpakatools::uniform_elements(acc, nClusters)) {
        const int begin_v = pfClusteringEdgeVars[v].mdpf_adjacencyIndex();
        const int end_v = pfClusteringEdgeVars[v + 1].mdpf_adjacencyIndex();
        int m = v;
        int i = begin_v;
        while ((m == v) && (i < end_v)) {
          m = alpaka::math::min(acc, m, pfClusteringEdgeVars[i].mdpf_adjacencyList());
          i++;
        }
        pfClusteringCCLabels[v].mdpf_topoId() = m;
      }
    }
  };

  class ECLCCLowDegreeComputeKernel {
  public:
    template <Idx default_mid_degree_threshold = 400>
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars) const {
      constexpr Idx mid_degree_threshold = default_mid_degree_threshold;

      constexpr int warpExtent = get_warp_size<Acc1D>();

      const unsigned int nClusters = pfClusteringCCLabels.size();

      const unsigned int low_degree_threshold = warpExtent / 2;  // also okay for warp size

      for (int v : ::cms::alpakatools::uniform_elements(acc, nClusters)) {
        if (pfClusteringCCLabels[v].mdpf_topoId() == v)
          continue;  // Skip if already its own representative

        const unsigned int begin_v = pfClusteringEdgeVars[v].mdpf_adjacencyIndex();
        const unsigned int end_v = pfClusteringEdgeVars[v + 1].mdpf_adjacencyIndex();
        const unsigned int deg_v = end_v - begin_v;

        if (deg_v > low_degree_threshold) {
          // Assign vertex to appropriate worklist based on degree
          int idx = (deg_v <= mid_degree_threshold)
                        ? alpaka::atomicAdd(acc, &pfClusteringCCLabels.topL(), +1, alpaka::hierarchy::Blocks{})
                        : alpaka::atomicAdd(acc, &pfClusteringCCLabels.topH(), -1, alpaka::hierarchy::Blocks{});
          pfClusteringCCLabels[idx].workl() = v;

          continue;
        }
        // Edge-process low-degree vertices:
        hook(acc, pfClusteringCCLabels, pfClusteringEdgeVars, v, begin_v, end_v, 1);
      }
    }
  };

  class ECLCCMidDegreeComputeKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars) const {
      const unsigned int nClusters = pfClusteringCCLabels.size();

      constexpr unsigned int w_extent = get_warp_size<Acc1D>();

      for (auto idx : ::cms::alpakatools::uniform_elements(acc, ::cms::alpakatools::round_up_by(nClusters, w_extent))) {
        const warp::warp_mask_t active_lanes_mask = alpaka::warp::ballot(acc, idx < nClusters);
        // Skip inactive lanes:
        if (idx >= nClusters)
          continue;

        const auto lane_idx = idx % w_extent;

        const auto x =
            lane_idx == 0 ? alpaka::atomicAdd(acc, &pfClusteringCCLabels.posL(), 1, alpaka::hierarchy::Blocks{}) : 0;

        int i = warp::shfl_mask(acc, active_lanes_mask, x, 0, w_extent);

        const int N = pfClusteringCCLabels.topL(); /*topL*/

        while (i < N) {
          const auto v = pfClusteringCCLabels[i].workl();

          const auto begin_v = pfClusteringEdgeVars[v].mdpf_adjacencyIndex() + lane_idx;
          const auto end_v = pfClusteringEdgeVars[v + 1].mdpf_adjacencyIndex();

          hook(acc, pfClusteringCCLabels, pfClusteringEdgeVars, v, begin_v, end_v, w_extent);
          // Assign the next vertex in the worklist
          const auto y =
              lane_idx == 0 ? alpaka::atomicAdd(acc, &pfClusteringCCLabels.posL(), 1, alpaka::hierarchy::Blocks{}) : 0;

          i = warp::shfl_mask(acc, active_lanes_mask, y, 0, w_extent);
        }
      }
    }
  };

  class ECLCCHighDegreeComputeKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc,
        reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels,
        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars) const {
      const unsigned int nClusters = pfClusteringCCLabels.size();

      auto const blockDim_x = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];

      int& v = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      const int topH = pfClusteringCCLabels.topH();

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nClusters)) {
          if (idx.local >= nClusters)
            continue;

          if (::cms::alpakatools::once_per_block(acc)) {
            int i = alpaka::atomicAdd(acc, &pfClusteringCCLabels.posH() /*posH*/, -1, alpaka::hierarchy::Grids{});
            v = (i > topH) ? pfClusteringCCLabels[i].workl() : -1;
          }
        }

        alpaka::syncBlockThreads(acc);

        while (v >= 0) {
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, nClusters)) {
            if (idx.local >= nClusters)
              continue;

            const int begin_v = pfClusteringEdgeVars[v].mdpf_adjacencyIndex() + idx.local;
            const int end_v = pfClusteringEdgeVars[v + 1].mdpf_adjacencyIndex();

            hook(acc, pfClusteringCCLabels, pfClusteringEdgeVars, v, begin_v, end_v, blockDim_x);

            if (::cms::alpakatools::once_per_block(acc)) {
              int i = alpaka::atomicAdd(acc, &pfClusteringCCLabels.posH() /*posH*/, -1, alpaka::hierarchy::Blocks{});
              v = (i > topH) ? pfClusteringCCLabels[i].workl() : -1;
            }
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    }
  };

  class ECLCCFlattenKernel {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View pfClusteringCCLabels) const {
      const unsigned int nClusters = pfClusteringCCLabels.size();

      for (int v : ::cms::alpakatools::uniform_elements(acc, nClusters)) {
        int vstat = pfClusteringCCLabels[v].mdpf_topoId();
        int next = pfClusteringCCLabels[vstat].mdpf_topoId();
        const int old = vstat;
        while (vstat > next) {
          vstat = next;
          next = pfClusteringCCLabels[vstat].mdpf_topoId();
        }
        if (old != vstat)
          pfClusteringCCLabels[v].mdpf_topoId() = vstat;
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc

#endif
