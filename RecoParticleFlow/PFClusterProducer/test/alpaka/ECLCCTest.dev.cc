/*
ECL-CC code: ECL-CC is a connected components graph algorithm. It operates
on graphs stored in binary CSR format.

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

#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <cstdlib>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterECLCC.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringCCLabelsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringEdgeVarsHostCollection.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ECLCCTest {
  public:
    void apply(Queue& queue,
               reco::PFMultiDepthClusteringCCLabelsDeviceCollection&,
               const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection&,
               const int) const;
  };

  void ECLCCTest::apply(Queue& queue,
                        reco::PFMultiDepthClusteringCCLabelsDeviceCollection& mdpfClusteringCCLabels,
                        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection& mdpfClusteringEdgeVars,
                        const int nClusters) const {
    uint32_t items = 960;

    uint32_t groups = cms::alpakatools::divide_up_by(nClusters, items);

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    // ECL-CC init stage:
    alpaka::exec<Acc1D>(
        queue, workDiv, eclcc::ECLCCInitKernel{}, mdpfClusteringCCLabels.view(), mdpfClusteringEdgeVars.view());
    // ECL-CC run low-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        eclcc::ECLCCLowDegreeComputeKernel{},
                        mdpfClusteringCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run mid-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        eclcc::ECLCCMidDegreeComputeKernel{},
                        mdpfClusteringCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run high-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        eclcc::ECLCCHighDegreeComputeKernel{},
                        mdpfClusteringCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run finalizing stage:
    alpaka::exec<Acc1D>(queue, workDiv, eclcc::ECLCCFlattenKernel{}, mdpfClusteringCCLabels.view());
    alpaka::wait(queue);
  }

  void launch_eclcc_test(Queue& queue,
                         ::reco::PFMultiDepthClusteringCCLabelsHostCollection& hostClusteringCCLabels,
                         const ::reco::PFMultiDepthClusteringEdgeVarsHostCollection& hostClusteringEdgeVars) {
    ECLCCTest eclcc_test{};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();

    const int nClusters = hClusteringCCLabels.size();

    reco::PFMultiDepthClusteringCCLabelsDeviceCollection devClusteringCCLabels{queue, nClusters};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection devClusteringEdgeVars{queue, 2 * nClusters};

    alpaka::memcpy(queue, devClusteringCCLabels.buffer(), hostClusteringCCLabels.buffer());
    alpaka::memcpy(queue, devClusteringEdgeVars.buffer(), hostClusteringEdgeVars.buffer());

    eclcc_test.apply(queue, devClusteringCCLabels, devClusteringEdgeVars, nClusters);

    alpaka::wait(queue);

    alpaka::memcpy(queue, hostClusteringCCLabels.buffer(), devClusteringCCLabels.buffer());

    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

std::vector<std::vector<int>> buildAdj(const std::vector<int>& v) {
  const int n = (int)v.size();
  std::vector<std::vector<int>> adj(n);
  for (int i = 0; i < n; ++i) {
    int k = v[i];
    if (0 <= k && k < n && k != i) {  //exclude self-loop
      adj[i].push_back(k);
      adj[k].push_back(i);
    }
  }
  for (auto& nbrs : adj) {
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }
  return adj;
}

void init(const int nodes,
          const int* const __restrict__ nidx,
          const int* const __restrict__ nlist,
          int* const __restrict__ nstat) {
  for (int v = 0; v < nodes; v++) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = std::min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }
}

static inline int representative(const int idx, int* const __restrict__ nstat) {
  int curr = nstat[idx];

  if (curr == idx)
    return curr;

  int prev = idx;
  int next = nstat[curr];  //nstat[nstat[idx]]

  while (curr > next) {
    nstat[prev] = next;  //now 'prev' points to 'next' vertex id
    prev = curr;
    curr = next;
    next = nstat[curr];
  }
  return curr;
}

void compute(const int nodes,
             const int* const __restrict__ nidx,
             const int* const __restrict__ nlist,
             int* const __restrict__ nstat) {
  for (int v = 0; v < nodes; v++) {
    int vstat = nstat[v];  //get parent

    if (v == vstat)
      continue;  //nop if root

    const int beg = nidx[v];
    const int end = nidx[v + 1];

    vstat = representative(v, nstat);  // get lowest vertex id that is connected to v (representative)

    for (int i = beg; i < end; i++) {
      const int nli = nlist[i];  // get neighbor id

      if (v <= nli)
        continue;  // nop if v is lower then neighbor nli (or coincide)

      int ostat =
          representative(nli, nstat);  // otherwise get neighbor' representative : minimal vertex connected to neigh

      bool repeat;

      do {
        repeat = false;

        if (vstat == ostat)
          continue;

        if (vstat < ostat) {
          //int ret = __sync_val_compare_and_swap(&nstat[ostat], ostat, vstat);
          int ret = nstat[ostat];
          if (ret == ostat) {
            nstat[ostat] = vstat;
          } else {
            ostat = ret;
            repeat = true;
          }
        } else {
          //int ret = __sync_val_compare_and_swap(&nstat[vstat], vstat, ostat);
          int ret = nstat[vstat];
          if (ret == vstat) {
            nstat[vstat] = ostat;
          } else {
            vstat = ret;
            repeat = true;
          }
        }
      } while (repeat);
    }
  }
}

void flatten(const int nodes, int* const __restrict__ nstat) {
  for (int v = 0; v < nodes; v++) {
    int vstat = nstat[v];
    const int old = vstat;
    int next = nstat[vstat];

    while (vstat > next) {
      vstat = next;
      next = nstat[vstat];
    }

    if (old != vstat)
      nstat[v] = vstat;
  }
}

void create(::reco::PFMultiDepthClusteringEdgeVarsHostCollection& hostClusteringEdgeVars,
            const std::vector<int>& roots,
            const int nClusters) {
  auto hClusteringEdgeVars = hostClusteringEdgeVars.view();

  std::mt19937 rng(12345);

  std::uniform_int_distribution<int> root0_degree_distr(1, 14);
  std::uniform_int_distribution<int> topo_distr(0, 8);

  std::vector<int> vx(nClusters, 0);

  int root0_max_degree = root0_degree_distr(rng);

  for (int i = 0; i < nClusters; ++i) {
    bool is_root = std::binary_search(roots.begin(), roots.end(), i);

    int vx_;

    if (is_root) {
      vx_ = i;
    } else {
      vx_ = topo_distr(rng);
      if (vx_ == 0) {
        if (root0_max_degree > 0)
          root0_max_degree -= 1;
        else
          vx_ += (topo_distr(rng) + 1);
      }
    }
    vx[i] = vx_;
  }

  auto adj = buildAdj(vx);

  int idx = 0;
  int mx_degree = 0;
  int mn_degree = 1024;
  for (int i = 0; i < (int)adj.size(); ++i) {
    hClusteringEdgeVars[i].mdpf_adjacencyIndex() = idx;

    if (mx_degree < (int)adj[i].size())
      mx_degree = (int)adj[i].size();
    if (mn_degree > (int)adj[i].size() && (int)adj[i].size() > 1)
      mn_degree = (int)adj[i].size();

    for (int j = 0; j < (int)adj[i].size(); ++j) {
      hClusteringEdgeVars[idx++].mdpf_adjacencyList() = adj[i][j];
    }
  }
  hClusteringEdgeVars[nClusters].mdpf_adjacencyIndex() = idx;
  printf("Max degree : %d;  Min degree %d\n", mx_degree, mn_degree);
}

int checkECLCC(const ::reco::PFMultiDepthClusteringCCLabelsHostCollection& hostClusteringCCLabels,
               const ::reco::PFMultiDepthClusteringEdgeVarsHostCollection& hostClusteringEdgeVars,
               const int nClusters) {
  auto hClusteringCCLabels = hostClusteringCCLabels.view();
  auto hClusteringEdgeVars = hostClusteringEdgeVars.view();

  const auto neigh = hClusteringEdgeVars[nClusters].mdpf_adjacencyIndex();

  std::vector<int> idx(nClusters + 1, 0);
  std::vector<int> adj(neigh, 0);

  for (int i = 0; i <= nClusters; i++)
    idx[i] = hClusteringEdgeVars[i].mdpf_adjacencyIndex();
  for (int i = 0; i < neigh; i++)
    adj[i] = hClusteringEdgeVars[i].mdpf_adjacencyList();

  // Run CC algo on the host:
  std::vector<int> status(nClusters);

  std::cout << "Run CPU ECL : " << std::endl;

  init(nClusters, idx.data(), adj.data(), status.data());
  compute(nClusters, idx.data(), adj.data(), status.data());
  flatten(nClusters, status.data());

  int nerrors = 0;

  int vidx = 0;
  for (auto& i : status) {
    const bool is_same = i == hClusteringCCLabels[vidx].mdpf_topoId();

    if (is_same == false) {
      printf("Error for TOPO %d != %d \t (%d) \n", hClusteringCCLabels[vidx].mdpf_topoId(), i, vidx);
      nerrors += 1;
    }
    vidx += 1;
  }
  return nerrors;
}

using namespace edm;
using namespace std;

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const int nClusters = 950;

  std::vector<int> roots = {0, 2, 3, 5, 7};

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    ::reco::PFMultiDepthClusteringCCLabelsHostCollection hostClusteringCCLabels{queue, nClusters};
    ::reco::PFMultiDepthClusteringEdgeVarsHostCollection hostClusteringEdgeVars{queue, 2 * nClusters};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();
    hClusteringCCLabels.size() = nClusters;

    create(hostClusteringEdgeVars, roots, nClusters);

    launch_eclcc_test(queue, hostClusteringCCLabels, hostClusteringEdgeVars);

    auto nerrs = checkECLCC(hostClusteringCCLabels, hostClusteringEdgeVars, nClusters);

    if (nerrs != 0) {
      std::cerr << nerrs << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
