#ifndef DataFormats_SiPixelClusterSoA_test_alpaka_Clusters_test_h
#define DataFormats_SiPixelClusterSoA_test_alpaka_Clusters_test_h

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testClusterSoA {

  void runKernels(SiPixelClustersSoAView clust_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testClusterSoA

#endif  // DataFormats_SiPixelClusterSoA_test_alpaka_Clusters_test_h
