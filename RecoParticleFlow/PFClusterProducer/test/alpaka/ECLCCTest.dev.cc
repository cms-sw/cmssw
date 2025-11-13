#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

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

using namespace reco;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  //using namespace cms::eclcc;

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
    uint32_t items = 224;

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

    reco::PFMultiDepthClusteringCCLabelsDeviceCollection devClusteringCCLabels{nClusters, queue};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection devClusteringEdgeVars{2 * nClusters, queue};

    alpaka::memcpy(queue, devClusteringCCLabels.buffer(), hostClusteringCCLabels.buffer());
    alpaka::memcpy(queue, devClusteringEdgeVars.buffer(), hostClusteringEdgeVars.buffer());

    eclcc_test.apply(queue, devClusteringCCLabels, devClusteringEdgeVars, nClusters);

    alpaka::wait(queue);

    alpaka::memcpy(queue, hostClusteringCCLabels.buffer(), devClusteringCCLabels.buffer());

    alpaka::wait(queue);

    for (int i = 0; i < nClusters; i++) {
      printf("TOPO  %d \t (%d) \n", hClusteringCCLabels[i].mdpf_topoId(), i);
    }
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
          int ret = __sync_val_compare_and_swap(&nstat[ostat], ostat, vstat);
          if (ret != ostat) {
            ostat = ret;
            repeat = true;
          }
        } else {
          int ret = __sync_val_compare_and_swap(&nstat[vstat], vstat, ostat);
          if (ret != vstat) {
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

  std::uniform_int_distribution<int> topo_distr(0, nClusters / 2 - 1);

  std::vector<int> vx(nClusters, 0);

  for (int i = 0; i < nClusters; ++i) {
    bool is_root = std::binary_search(roots.begin(), roots.end(), i);

    vx[i] = is_root ? i : topo_distr(rng);
  }

  auto adj = buildAdj(vx);

  int idx = 0;
  for (int i = 0; i < (int)adj.size(); ++i) {
    hClusteringEdgeVars[i].mdpf_adjacencyIndex() = idx;
    for (int j = 0; j < (int)adj[i].size(); ++j) {
      hClusteringEdgeVars[idx++].mdpf_adjacencyList() = adj[i][j];
    }
  }
  hClusteringEdgeVars[nClusters].mdpf_adjacencyIndex() = idx;  //
}

void check(const ::reco::PFMultiDepthClusteringEdgeVarsHostCollection& hostClusteringEdgeVars, const int nClusters) {
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

  std::cout << "ECL : " << std::endl;

  init(nClusters, idx.data(), adj.data(), status.data());
  compute(nClusters, idx.data(), adj.data(), status.data());
  flatten(nClusters, status.data());

  for (auto& i : status)
    std::cout << i << std::endl;
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

  const int nClusters = 200;

  std::vector<int> roots = {0, 3, 7, 11, 19, 29, 37, 41, 71, 83, 97};

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    ::reco::PFMultiDepthClusteringCCLabelsHostCollection hostClusteringCCLabels{nClusters, queue};
    ::reco::PFMultiDepthClusteringEdgeVarsHostCollection hostClusteringEdgeVars{2 * nClusters, queue};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();
    hClusteringCCLabels.size() = nClusters;

    create(hostClusteringEdgeVars, roots, nClusters);

    launch_eclcc_test(queue, hostClusteringCCLabels, hostClusteringEdgeVars);

    check(hostClusteringEdgeVars, nClusters);
  }

  return 0;
}
