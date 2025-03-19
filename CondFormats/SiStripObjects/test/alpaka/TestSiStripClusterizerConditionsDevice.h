#ifndef CondFormats_SiStripObjects_test_alpaka_TestSiStripClusterizerConditionsDevice_h
#define CondFormats_SiStripObjects_test_alpaka_TestSiStripClusterizerConditionsDevice_h

#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testConditionsSoA {

  void runKernels(SiStripClusterizerConditionsDetToFedsView DetToFedsView,
                  SiStripClusterizerConditionsData_fedchView Data_fedchView,
                  SiStripClusterizerConditionsData_stripView Data_stripView,
                  SiStripClusterizerConditionsData_apvView Data_apvView,
                  Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testConditionsSoA

#endif  // CondFormats_SiStripObjects_test_alpaka_TestSiStripClusterizerConditionsDevice_h
