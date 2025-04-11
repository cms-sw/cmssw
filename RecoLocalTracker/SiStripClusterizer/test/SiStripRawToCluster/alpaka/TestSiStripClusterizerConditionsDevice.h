#ifndef RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_TestSiStripClusterizerConditionsDevice_h
#define RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_TestSiStripClusterizerConditionsDevice_h

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testConditionsSoA {
  using namespace ::sistrip;
  void runKernels(SiStripClusterizerConditionsDetToFedsView DetToFedsView,
                  SiStripClusterizerConditionsData_fedchView Data_fedchView,
                  SiStripClusterizerConditionsData_stripView Data_stripView,
                  SiStripClusterizerConditionsData_apvView Data_apvView,
                  Queue& queue);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip::testConditionsSoA

#endif  // RecoLocalTracker_SiStripClusterizer_test_SiStripRawToCluster_TestSiStripClusterizerConditionsDevice_h
