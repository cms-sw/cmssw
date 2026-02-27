#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <cstdlib>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

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

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCPrologue.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringCCLabelsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringEdgeVarsHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PrologueTest {
  public:
    void apply(Queue &queue,
               reco::PFMultiDepthClusteringEdgeVarsDeviceCollection &pfClusteringEdgeVars,
               const reco::PFMultiDepthClusteringCCLabelsDeviceCollection &mdpfClusteringVars) const;
  };

  void PrologueTest::apply(Queue &queue,
                           reco::PFMultiDepthClusteringEdgeVarsDeviceCollection &pfClusteringEdgeVars,
                           const reco::PFMultiDepthClusteringCCLabelsDeviceCollection &mdpfClusteringVars) const {
    uint32_t items = 160;

    auto n = static_cast<uint32_t>(mdpfClusteringVars->metadata().size());
    uint32_t groups = cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    } else {
      printf("Number of groups :: %d\n", groups);
    }

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, ECLCCPrologueKernel{}, pfClusteringEdgeVars.view(), mdpfClusteringVars.view());

    alpaka::wait(queue);
  }

  void launch_prologue_test(Queue &queue,
                            ::reco::PFMultiDepthClusteringEdgeVarsHostCollection &hostClusteringEdgeVars,
                            const ::reco::PFMultiDepthClusteringCCLabelsHostCollection &hostClusteringCCLabels) {
    PrologueTest prologue_test{};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();

    const int nClusters = hClusteringCCLabels.size();

    reco::PFMultiDepthClusteringCCLabelsDeviceCollection devClusteringCCLabels{queue, nClusters};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection devClusteringEdgeVars{queue, 2 * nClusters};

    alpaka::memcpy(queue, devClusteringCCLabels.buffer(), hostClusteringCCLabels.buffer());

    prologue_test.apply(queue, devClusteringEdgeVars, devClusteringCCLabels);

    alpaka::wait(queue);

    alpaka::memcpy(queue, hostClusteringEdgeVars.buffer(), devClusteringEdgeVars.buffer());

    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

std::vector<std::vector<int>> buildAdj(const std::vector<int> &v) {
  const int n = (int)v.size();
  std::vector<std::vector<int>> adj(n);
  for (int i = 0; i < n; ++i) {
    int k = v[i];
    if (0 <= k && k < n && k != i) {  //exclude self-loop
      adj[i].push_back(k);
      adj[k].push_back(i);
    }
  }
  // de-duplicate and sort each neighbor list
  for (auto &nbrs : adj) {
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }
  return adj;
}

void create(::reco::PFMultiDepthClusteringCCLabelsHostCollection &hostClusteringCCLabels,
            const std::vector<int> &roots,
            const int nClusters) {
  auto hClusteringCCLabels = hostClusteringCCLabels.view();

  std::mt19937 rng(12345);

  std::uniform_int_distribution<int> topo_distr(0, nClusters - 1);

  std::vector<int> vx(nClusters, 0);

  for (int i = 0; i < nClusters; ++i) {
    bool is_root = std::binary_search(roots.begin(), roots.end(), i);

    vx[i] = is_root ? i : topo_distr(rng);
  }

  for (int i = 0; i < nClusters; ++i) {
    const int j = vx[i];
    hClusteringCCLabels[i].mdpf_topoId() = (vx[j] == i && j < i) ? i : vx[i];
  }
}

int checkPrologue(const ::reco::PFMultiDepthClusteringEdgeVarsHostCollection &hostClusteringEdgeVars,
                  const ::reco::PFMultiDepthClusteringCCLabelsHostCollection &hostClusteringCCLabels,
                  const int nClusters) {
  auto hClusteringCCLabels = hostClusteringCCLabels.view();

  std::vector<int> vx(nClusters, 0);

  for (int i = 0; i < nClusters; ++i) {
    vx[i] = hClusteringCCLabels[i].mdpf_topoId();
  }

  auto adj = buildAdj(vx);

  auto hClusteringEdgeVars = hostClusteringEdgeVars.view();

  int nerrors = 0;

  for (int i = 0; i < nClusters; ++i) {
    const int begin = hClusteringEdgeVars[i].mdpf_adjacencyIndex();
    const int end = hClusteringEdgeVars[i + 1].mdpf_adjacencyIndex();

    if ((end - begin) != static_cast<int>(adj[i].size()))
      std::cout << "Vertex degree mismatch " << i << " Legacy size " << static_cast<int>(adj[i].size())
                << " Alpaka size " << (end - begin) << std::endl;

    std::cout << i << ": Legacy [";
    for (int j = 0; j < (int)adj[i].size(); ++j) {
      if (j)
        std::cout << ", ";
      std::cout << adj[i][j];
    }
    std::cout << "]   ";

    std::cout << "\t\t Alpaka [";
    for (int j = begin; j < end; j++) {
      const int idx = hClusteringEdgeVars[j].mdpf_adjacencyList();
      const bool is_found = std::binary_search(adj[i].begin(), adj[i].end(), idx);

      if (is_found == false) {
        nerrors += 1;
        std::cout << "mismatch detected for vertex " << i << ", index  " << idx << std::endl;
      }

      std::cout << hClusteringEdgeVars[j].mdpf_adjacencyList() << " ";
    }
    std::cout << "]\n";
  }

  return nerrors;
}

using namespace edm;
using namespace std;

int main() {
  // get the list of devices on the current platform
  auto const &devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const int nClusters = 145;

  std::vector<int> roots = {0, 3, 7, 11, 19, 29, 37, 41, 71, 83, 97, 101, 137};

  // run the test on each device
  for (auto const &device : devices) {
    auto queue = Queue(device);

    ::reco::PFMultiDepthClusteringCCLabelsHostCollection hostClusteringCCLabels{queue, nClusters};
    ::reco::PFMultiDepthClusteringEdgeVarsHostCollection hostClusteringEdgeVars{queue, 2 * nClusters};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();
    hClusteringCCLabels.size() = nClusters;

    create(hostClusteringCCLabels, roots, nClusters);

    launch_prologue_test(queue, hostClusteringEdgeVars, hostClusteringCCLabels);

    auto nerrors = checkPrologue(hostClusteringEdgeVars, hostClusteringCCLabels, nClusters);

    if (nerrors != 0) {
      std::cerr << nerrors << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
