#ifndef CondFormats_SiStripObjects_test_alpaka_TestSiStripMappingDevice_h
#define CondFormats_SiStripObjects_test_alpaka_TestSiStripMappingDevice_h

#include "CondFormats/SiStripObjects/interface/SiStripMappingSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testMappingSoA {

  void runKernels(SiStripMappingView clust_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testMappingSoA

#endif  // CondFormats_SiStripObjects_test_alpaka_TestSiStripMappingDevice_h
