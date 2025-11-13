#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

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

using namespace reco;

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
    uint32_t items = 128;

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

    reco::PFMultiDepthClusteringCCLabelsDeviceCollection devClusteringCCLabels{nClusters, queue};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection devClusteringEdgeVars{2 * nClusters, queue};

    alpaka::memcpy(queue, devClusteringCCLabels.buffer(), hostClusteringCCLabels.buffer());

    prologue_test.apply(queue, devClusteringEdgeVars, devClusteringCCLabels);

    alpaka::wait(queue);

    alpaka::memcpy(queue, hostClusteringEdgeVars.buffer(), devClusteringEdgeVars.buffer());

    auto hClusteringEdgeVars = hostClusteringEdgeVars.view();

    alpaka::wait(queue);

    for (int i = 0; i < nClusters; i++) {
      const int begin = hClusteringEdgeVars[i].mdpf_adjacencyIndex();
      const int end = hClusteringEdgeVars[i + 1].mdpf_adjacencyIndex();

      std::cout << i << ": [";
      for (int j = begin; j < end; j++) {
        std::cout << hClusteringEdgeVars[j].mdpf_adjacencyList() << " ";
      }
      std::cout << "]\n";
    }
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
    printf("FROM = %d \t: TO = %d\n", vx[i], i);
  }

  for (int i = 0; i < nClusters; ++i) {
    hClusteringCCLabels[i].mdpf_topoId() = vx[i];
  }

  auto adj = buildAdj(vx);

  for (int i = 0; i < (int)adj.size(); ++i) {
    std::cout << i << ": [";
    for (int j = 0; j < (int)adj[i].size(); ++j) {
      if (j)
        std::cout << ", ";
      std::cout << adj[i][j];
    }
    std::cout << "]\n";
  }
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

  const int nClusters = 100;

  std::vector<int> roots = {0, 3, 7, 11, 19, 29, 37, 41, 71, 83, 97};

  // run the test on each device
  for (auto const &device : devices) {
    auto queue = Queue(device);

    ::reco::PFMultiDepthClusteringCCLabelsHostCollection hostClusteringCCLabels{nClusters, queue};
    ::reco::PFMultiDepthClusteringEdgeVarsHostCollection hostClusteringEdgeVars{2 * nClusters, queue};

    auto hClusteringCCLabels = hostClusteringCCLabels.view();
    hClusteringCCLabels.size() = nClusters;

    create(hostClusteringCCLabels, roots, nClusters);

    launch_prologue_test(queue, hostClusteringEdgeVars, hostClusteringCCLabels);
  }

  return 0;
}
