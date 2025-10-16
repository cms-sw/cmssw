#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterECLCC_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterECLCC_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"

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

  using namespace cms::alpakatools;

  // Algorithm internal data:
  template <typename TBufAcc, Idx default_mid_degree_threshold = 400>
  class CCGAlgorithmArgs {
  public:
    static constexpr Idx mid_degree_threshold = default_mid_degree_threshold;

    reco::PFMultiDepthClusteringVarsDeviceCollection::View pfClusteringVars_;
    const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars_;

    int* workl;

    int* tp;

    const unsigned int nClusters;

    CCGAlgorithmArgs(Queue& queue,
                     reco::PFMultiDepthClusteringVarsDeviceCollection& pfClusteringVars,
                     const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection& pfClusteringEdgeVars,
                     TBufAcc& workl,
                     TBufAcc& tp,
                     const unsigned int nClusters_ = 0)
        : pfClusteringVars_(pfClusteringVars.view()),
          pfClusteringEdgeVars_(pfClusteringEdgeVars.view()),
          workl(workl.data()),
          tp(tp.data()),
          nClusters(nClusters_) {}
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE static void hook(
      TAcc const& acc,
      reco::PFMultiDepthClusteringVarsDeviceCollection::View pfClusteringVars,
      const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView pfClusteringEdgeVars,
      const int v,
      const int begin_v,
      const int end_v,
      const int offset_v) {
    auto representative = [pfClusteringVarsPtr = &pfClusteringVars](const int v) -> int {
      int curr_v = (*pfClusteringVarsPtr)[v].mdpf_topoId();

      if (curr_v == v)
        return curr_v;

      int prev_v = v;
      int next_v = (*pfClusteringVarsPtr)[curr_v].mdpf_topoId();

      while (curr_v > next_v) {
        (*pfClusteringVarsPtr)[prev_v].mdpf_topoId() = next_v;
        prev_v = curr_v;
        curr_v = next_v;
        next_v = (*pfClusteringVarsPtr)[curr_v].mdpf_topoId();
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

        int tmp = alpaka::atomicCas(acc, &pfClusteringVars[high_rep].mdpf_topoId(), high_rep, low_rep);

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
    template <typename TAcc, typename TArgs>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TArgs* args_ptr) const {
      TArgs& args = *args_ptr;
      for (int v : ::cms::alpakatools::uniform_elements(acc, args.nClusters)) {
        const int begin_v = args.pfClusteringEdgeVars_[v].mdpf_adjacencyIndex();
        const int end_v = args.pfClusteringEdgeVars_[v + 1].mdpf_adjacencyIndex();
        int m = v;
        int i = begin_v;
        while ((m == v) && (i < end_v)) {
          m = alpaka::math::min(acc, m, args.pfClusteringEdgeVars_[i].mdpf_adjacencyList());
          i++;
        }
        args.pfClusteringVars_[v].mdpf_topoId() = m;

        if (v == 0) {
          args.tp[2] = args.nClusters - 1;  //topH
          args.tp[3] = args.nClusters - 1;  //posH
          args.pfClusteringVars_.size() = args.nClusters;
        }
      }
    }
  };

  class ECLCCLowDegreeComputeKernel {
  public:
    template <typename TAcc, typename TArgs>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TArgs* args_ptr) const {
      TArgs& args = *args_ptr;

      const int warpExtent = alpaka::warp::getSize(acc);

      const unsigned int low_degree_threshold = warpExtent / 2;  // also okay for warp size

      for (int v : ::cms::alpakatools::uniform_elements(acc, args.nClusters)) {
        if (args.pfClusteringVars_[v].mdpf_topoId() == v)
          continue;  // Skip if already its own representative

        const unsigned int begin_v = args.pfClusteringEdgeVars_[v].mdpf_adjacencyIndex();
        const unsigned int end_v = args.pfClusteringEdgeVars_[v + 1].mdpf_adjacencyIndex();
        const unsigned int deg_v = end_v - begin_v;

        if (deg_v > low_degree_threshold) {
          // Assign vertex to appropriate worklist based on degree
          int idx = (deg_v <= TArgs::mid_degree_threshold)
                        ? alpaka::atomicAdd(acc, &args.tp[0] /*topL*/, +1, alpaka::hierarchy::Blocks{})
                        : alpaka::atomicAdd(acc, &args.tp[2] /*topH*/, -1, alpaka::hierarchy::Blocks{});
          args.workl[idx] = v;

          continue;
        }
        // Edge-process low-degree vertices:
        hook(acc, args.pfClusteringVars_, args.pfClusteringEdgeVars_, v, begin_v, end_v, 1);
      }
    }
  };

  class ECLCCMidDegreeComputeKernel {
  public:
    template <typename TAcc, typename TArgs>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TArgs* args_ptr) const {
      TArgs& args = *args_ptr;

      const int w_extent = alpaka::warp::getSize(acc);

      for (auto idx :
           ::cms::alpakatools::uniform_elements(acc, ::cms::alpakatools::round_up_by(args.nClusters, w_extent))) {
        const unsigned int active_lanes_mask = alpaka::warp::ballot(acc, idx < args.nClusters);
        // Skip inactive lanes:
        if (idx >= args.nClusters)
          continue;

        const auto lane_idx = idx % w_extent;

        const auto x = lane_idx == 0 ? alpaka::atomicAdd(acc, &(args.tp[1]), 1, alpaka::hierarchy::Blocks{}) : 0;

        warp::syncWarpThreads_mask(acc, active_lanes_mask);

        int i = warp::shfl_mask(acc, active_lanes_mask, x, 0, w_extent);

        const int N = args.tp[0]; /*topL*/

        while (i < N) {
          const auto v = args.workl[i];

          const auto begin_v = args.pfClusteringEdgeVars_[v].mdpf_adjacencyIndex() + lane_idx;
          const auto end_v = args.pfClusteringEdgeVars_[v + 1].mdpf_adjacencyIndex();

          hook(acc, args.pfClusteringVars_, args.pfClusteringEdgeVars_, v, begin_v, end_v, w_extent);
          // Assign the next vertex in the worklist
          const auto y = lane_idx == 0 ? alpaka::atomicAdd(acc, &(args.tp[1]), 1, alpaka::hierarchy::Blocks{}) : 0;

          warp::syncWarpThreads_mask(acc, active_lanes_mask);

          i = warp::shfl_mask(acc, active_lanes_mask, y, 0, w_extent);
        }
      }
    }
  };

  class ECLCCHighDegreeComputeKernel {
  public:
    template <typename TAcc, typename TArgs>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TArgs* args_ptr) const {
      TArgs& args = *args_ptr;

      //using data_t = typename TArgs::data_t;

      auto const blockDim_x = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];

      int& v = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      const int topH = args.tp[2];

      for (auto group : ::cms::alpakatools::uniform_groups(acc)) {
        for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, args.nClusters)) {
          if (idx.local >= args.nClusters)
            continue;

          if (::cms::alpakatools::once_per_block(acc)) {
            int i = alpaka::atomicAdd(acc, static_cast<int*>(&args.tp[3]) /*posH*/, -1, alpaka::hierarchy::Grids{});
            v = (i > topH) ? args.workl[i] : -1;
          }
        }

        alpaka::syncBlockThreads(acc);

        while (v >= 0) {
          for (auto idx : ::cms::alpakatools::uniform_group_elements(acc, group, args.nClusters)) {
            if (idx.local >= args.nClusters)
              continue;

            const int begin_v = args.pfClusteringEdgeVars_[v].mdpf_adjacencyIndex() + idx.local;
            const int end_v = args.pfClusteringEdgeVars_[v + 1].mdpf_adjacencyIndex();

            hook(acc, args.pfClusteringVars_, args.pfClusteringEdgeVars_, v, begin_v, end_v, blockDim_x);

            if (::cms::alpakatools::once_per_block(acc)) {
              int i = alpaka::atomicAdd(acc, static_cast<int*>(&args.tp[3]) /*posH*/, -1, alpaka::hierarchy::Blocks{});
              v = (i > topH) ? args.workl[i] : -1;
            }
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    }
  };

  class ECLCCFlattenKernel {
  public:
    template <typename TAcc, typename TArgs>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TArgs* args_ptr) const {
      TArgs& args = *args_ptr;
      for (int v : ::cms::alpakatools::uniform_elements(acc, args.nClusters)) {
        int vstat = args.pfClusteringVars_[v].mdpf_topoId();
        int next = args.pfClusteringVars_[vstat].mdpf_topoId();
        const int old = vstat;
        while (vstat > next) {
          vstat = next;
          next = args.pfClusteringVars_[vstat].mdpf_topoId();
        }
        if (old != vstat)
          args.pfClusteringVars_[v].mdpf_topoId() = vstat;
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc

#endif
