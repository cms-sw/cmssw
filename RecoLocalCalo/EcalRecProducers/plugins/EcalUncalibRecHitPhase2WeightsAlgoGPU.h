#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h

#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"

#include "DeclsForKernelsPh2.h"

namespace ecal {
  namespace weights {

    void entryPoint(ecal::DigisCollection<calo::common::DevStoragePolicy> const&,
                    EventOutputDataGPU&,
                    cms::cuda::device::unique_ptr<double[]>&,
                    cudaStream_t);

  }  // namespace weights
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoGPU_h
