#ifndef DataFormats_SiPixelDigiSoA_test_alpaka_Digis_test_h
#define DataFormats_SiPixelDigiSoA_test_alpaka_Digis_test_h

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA {

  void runKernels(SiPixelDigisSoAView digis_view, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA

#endif  // DataFormats_SiPixelDigiSoA_test_alpaka_Digis_test_h
