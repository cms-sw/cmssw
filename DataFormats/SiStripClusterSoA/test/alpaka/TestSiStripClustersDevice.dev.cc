#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestSiStripClustersDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiStripClusterSoA {
  using namespace sistrip;
  using namespace cms::alpakatools;

  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripClustersView clust_view,
                                  uint32_t nSeedStripsNC,
                                  uint32_t clusterSizeLimit) const {
      for (int32_t j : cms::alpakatools::uniform_elements(acc, clust_view.metadata().size())) {
        clust_view.clusterIndex(j) = (uint32_t)(j);
        clust_view.clusterSize(j) = (uint32_t)(j * 2);
        for (int k = 0; k < maxStripsPerCluster; ++k) {
          clust_view.clusterADCs(j)[k] = (uint8_t)((j + k) % 255);
        }
        clust_view.clusterDetId(j) = (uint32_t)(j + 12);
        clust_view.firstStrip(j) = (uint16_t)(j % 65536);
        clust_view.trueCluster(j) = (bool)((j % 2 == 0));
        clust_view.barycenter(j) = (float)(j * 1.0f);
        clust_view.charge(j) = (float)(j * -1.0f);
      }

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        clust_view.nClusters() = nSeedStripsNC;
        clust_view.maxClusterSize() = clusterSizeLimit;
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripClustersConstView clust_view,
                                  uint32_t nSeedStripsNC,
                                  uint32_t clusterSizeLimit) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, clust_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(clust_view[j].clusterIndex() == j);
        ALPAKA_ASSERT_ACC(clust_view[j].clusterSize() == j * 2);
        for (int k = 0; k < maxStripsPerCluster; ++k) {
          ALPAKA_ASSERT_ACC(clust_view.clusterADCs(j)[k] == (uint8_t)((j + k) % 255));
        }
        ALPAKA_ASSERT_ACC(clust_view[j].clusterDetId() == j + 12);
        ALPAKA_ASSERT_ACC(clust_view[j].firstStrip() == j % 65536);
        ALPAKA_ASSERT_ACC(clust_view[j].trueCluster() == (j % 2 == 0));
        ALPAKA_ASSERT_ACC(clust_view[j].barycenter() == j * 1.0f);
        ALPAKA_ASSERT_ACC(clust_view[j].charge() == j * -1.0f);
      }
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        ALPAKA_ASSERT_ACC(clust_view.nClusters() == nSeedStripsNC);
        ALPAKA_ASSERT_ACC(clust_view.maxClusterSize() == clusterSizeLimit);
      }
    }
  };

  void runKernels(SiStripClustersView clust_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(clust_view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, clust_view, kMaxSeedStrips, maxStripsPerCluster);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, clust_view, kMaxSeedStrips, maxStripsPerCluster);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiStripClusterSoA
