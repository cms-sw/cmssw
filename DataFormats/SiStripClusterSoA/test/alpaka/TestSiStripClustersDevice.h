#ifndef DataFormats_SiStripClusterSoA_test_alpaka_TestSiStripClustersDevice_h
#define DataFormats_SiStripClusterSoA_test_alpaka_TestSiStripClustersDevice_h

#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiStripClusterSoA {
  constexpr int kMaxSeedStrips = 200000;

  void runKernels(sistrip::SiStripClustersView clust_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiStripClusterSoA

#endif  // DataFormats_SiStripClusterSoA_test_alpaka_TestSiStripClustersDevice_h
