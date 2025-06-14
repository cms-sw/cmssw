#ifndef RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_alpaka_TestSiStripMappingDevice_h
#define RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_alpaka_TestSiStripMappingDevice_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testMappingSoA {
  using namespace ::sistrip;
  void runKernels(SiStripMappingView clust_view, Queue& queue);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testMappingSoA

#endif  // RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_alpaka_TestSiStripMappingDevice_h
