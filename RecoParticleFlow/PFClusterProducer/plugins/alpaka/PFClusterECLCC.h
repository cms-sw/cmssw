#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFClusterECLCC_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFClusterECLCC_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusteringEdgeVarsDeviceCollection.h"

// The following comment block is required in using the ECL-CC algorithm for topological clustering

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

/*
 The code is modified for the specific use-case of generating topological clusters
 for PFClustering. It is adapted to work with the Alpaka portability library. The
 kernels for processing vertices at warp and block level granularity have been
 removed since the degree of vertices in our inputs is only ever 8; the number of
 neighbors.
*/

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /* intermediate pointer jumping */

  ALPAKA_FN_ACC inline int representative(const int idx,
                                          reco::PFClusteringVarsDeviceCollection::View pfClusteringVars) {
    int curr = pfClusteringVars[idx].pfrh_topoId();
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = pfClusteringVars[curr].pfrh_topoId())) {
        pfClusteringVars[prev].pfrh_topoId() = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  // Initial step of ECL-CC. Uses ID of first neighbour in edgeList with a smaller ID
  class ECLCCInit {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        const int beg = pfClusteringEdgeVars[v].pfrh_edgeIdx();
        const int end = pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
        int m = v;
        int i = beg;
        while ((m == v) && (i < end)) {
          m = std::min(m, pfClusteringEdgeVars[i].pfrh_edgeList());
          i++;
        }
        pfClusteringVars[v].pfrh_topoId() = m;
      }
    }
  };

  // First edge processing kernel of ECL-CC
  // Processes vertices
  class ECLCCCompute1 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        const int vstat = pfClusteringVars[v].pfrh_topoId();
        if (v != vstat) {
          const int beg = pfClusteringEdgeVars[v].pfrh_edgeIdx();
          const int end = pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
          int vstat = representative(v, pfClusteringVars);
          for (int i = beg; i < end; i++) {
            const int nli = pfClusteringEdgeVars[i].pfrh_edgeList();
            if (v > nli) {
              int ostat = representative(nli, pfClusteringVars);
              bool repeat;
              do {
                repeat = false;
                if (vstat != ostat) {
                  int ret;
                  if (vstat < ostat) {
                    if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                      ostat = ret;
                      repeat = true;
                    }
                  } else {
                    if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                      vstat = ret;
                      repeat = true;
                    }
                  }
                }
              } while (repeat);
            }
          }
        }
      }
    }
  };

  /* link all vertices to sink */
  class ECLCCFlatten {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        int next, vstat = pfClusteringVars[v].pfrh_topoId();
        const int old = vstat;
        while (vstat > (next = pfClusteringVars[vstat].pfrh_topoId())) {
          vstat = next;
        }
        if (old != vstat)
          pfClusteringVars[v].pfrh_topoId() = vstat;
      }
    }
  };

  // ECL-CC ends

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
