#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMultiFitAlgo_gpu_new_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMultiFitAlgo_gpu_new_HH

#include <vector>

#include <cuda.h>

#include "RecoLocalCalo/EcalRecAlgos/interface/DeclsForKernels.h"

namespace ecal {
  namespace multifit {

    void entryPoint(EventInputDataGPU const&,
                    EventOutputDataGPU&,
                    EventDataForScratchGPU&,
                    ConditionsProducts const&,
                    ConfigurationParameters const&,
                    cudaStream_t);

  }
}  // namespace ecal

#endif
