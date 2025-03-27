#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsAlgoPortable_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitPhase2WeightsAlgoPortable_h

#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "EcalUncalibRecHitPhase2WeightsStruct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::weights {

  void phase2Weights(EcalDigiPhase2DeviceCollection const &digis,
                         EcalUncalibratedRecHitDeviceCollection &uncalibratedRecHits,
                         EcalUncalibRecHitPhase2Weights const* weightsObj,
			 Queue &queue);

}  //namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::weights

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsAlgoPortable_h
