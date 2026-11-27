#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFClusterECLCC_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFClusterECLCC_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusteringEdgeVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusteringVarsDeviceCollection.h"

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

  // Device-coherent accesses to pfrh_topoId: ECLCCCompute1 updates it with atomicCas while other threads read and
  // write it, and on GPUs whose caches are not coherent for plain accesses (e.g. the 8 XCDs of an MI300X) plain loads
  // can see stale values. The original ECL-CC code uses volatile accesses for the same reason.
  ALPAKA_FN_ACC inline int loadTopoId(Acc1D const& acc,
                                      reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                      const int idx) {
    return alpaka::atomicAdd(acc, &pfClusteringVars[idx].pfrh_topoId(), 0);
  }

  ALPAKA_FN_ACC inline void storeTopoId(Acc1D const& acc,
                                        reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                        const int idx,
                                        const int value) {
    alpaka::atomicExch(acc, &pfClusteringVars[idx].pfrh_topoId(), value);
  }

  ALPAKA_FN_ACC inline int representative(Acc1D const& acc,
                                          const int idx,
                                          reco::PFClusteringVarsDeviceCollection::View pfClusteringVars) {
    int curr = loadTopoId(acc, pfClusteringVars, idx);
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = loadTopoId(acc, pfClusteringVars, curr))) {
        storeTopoId(acc, pfClusteringVars, prev, next);
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  // Initial step of ECL-CC. Uses ID of first neighbour in edgeList with a smaller ID
  class ECLCCInit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      for (int v : cms::alpakatools::uniform_elements(acc, nRH)) {
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::uniform_elements(acc, nRH)) {
        const int vstat = loadTopoId(acc, pfClusteringVars, v);
        if (v != vstat) {
          const int beg = pfClusteringEdgeVars[v].pfrh_edgeIdx();
          const int end = pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
          int vstat = representative(acc, v, pfClusteringVars);
          for (int i = beg; i < end; i++) {
            const int nli = pfClusteringEdgeVars[i].pfrh_edgeList();
            if (v > nli) {
              int ostat = representative(acc, nli, pfClusteringVars);
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::uniform_elements(acc, nRH)) {
        int next, vstat = loadTopoId(acc, pfClusteringVars, v);
        const int old = vstat;
        while (vstat > (next = loadTopoId(acc, pfClusteringVars, vstat))) {
          vstat = next;
        }
        if (old != vstat)
          storeTopoId(acc, pfClusteringVars, v, vstat);
      }
    }
  };

  // ECL-CC ends

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
