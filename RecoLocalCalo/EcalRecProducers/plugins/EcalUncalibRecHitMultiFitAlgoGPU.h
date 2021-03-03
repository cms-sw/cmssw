#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitMultiFitAlgoGPU_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitMultiFitAlgoGPU_h

#include <vector>

#include <cuda.h>

#include "DeclsForKernels.h"

namespace ecal {
  namespace multifit {

    void entryPoint(EventInputDataGPU const&,
                    EventOutputDataGPU&,
                    EventDataForScratchGPU&,
                    ConditionsProducts const&,
                    ConfigurationParameters const&,
                    cudaStream_t);

  }  // namespace multifit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitMultiFitAlgoGPU_h
