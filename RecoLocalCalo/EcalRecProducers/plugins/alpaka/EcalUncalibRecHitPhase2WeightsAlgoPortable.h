#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsAlgoPortable_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsAlgoPortable_h

#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace ecal {
    namespace weights {

      void phase2Weights(EcalDigiPhase2DeviceCollection const &digis,
                         EcalUncalibratedRecHitDeviceCollection &uncalibratedRecHits,
                         const cms::alpakatools::host_buffer<double[]> &weights,
                         const cms::alpakatools::host_buffer<double[]> &timeWeights,
                         Queue &queue);

    }  // namespace weights
  }  // namespace ecal
}  //namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoPortable_h
