#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h

#include <vector>

#include <cuda.h>

#include "DeclsForKernelsPh2WeightsGPU.h"

namespace ecal {
  namespace weights {

    void entryPoint(ecal::DigisCollection<calo::common::DevStoragePolicy> const&,
                    EventOutputDataGPUWeights&,
                    cms::cuda::device::unique_ptr<double[]>& ,
                    // cms::cuda::device::unique_ptr<double[]>& ,
                    cudaStream_t);

  }  // namespace weights
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h
