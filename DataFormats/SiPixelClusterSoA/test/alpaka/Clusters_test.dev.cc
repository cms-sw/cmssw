#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Clusters_test.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::testClusterSoA {

  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelClustersSoAView clust_view) const {
      for (int32_t j : cms::alpakatools::uniform_elements(acc, clust_view.metadata().size())) {
        clust_view[j].moduleStart() = j;
        clust_view[j].clusInModule() = j * 2;
        clust_view[j].moduleId() = j * 3;
        clust_view[j].clusModuleStart() = j * 4;
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, SiPixelClustersSoAConstView clust_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, clust_view.metadata().size())) {
        assert(clust_view[j].moduleStart() == j);
        assert(clust_view[j].clusInModule() == j * 2);
        assert(clust_view[j].moduleId() == j * 3);
        assert(clust_view[j].clusModuleStart() == j * 4);
      }
    }
  };

  void runKernels(SiPixelClustersSoAView clust_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(clust_view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, clust_view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, clust_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testClusterSoA
