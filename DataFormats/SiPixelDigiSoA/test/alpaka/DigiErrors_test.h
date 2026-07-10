#ifndef DataFormats_SiPixelDigiSoA_test_alpaka_DigiErrors_test_h
#define DataFormats_SiPixelDigiSoA_test_alpaka_DigiErrors_test_h

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA {

  void runKernels(Queue& queue, SiPixelDigiErrorsSoAView digiErrors_view);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA

#endif  // DataFormats_SiPixelDigiSoA_test_alpaka_DigiErrors_test_h
