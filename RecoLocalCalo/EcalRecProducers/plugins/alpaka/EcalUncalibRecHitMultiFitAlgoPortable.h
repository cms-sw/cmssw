#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h

#include <vector>

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DeclsForKernels.h"
#include "EcalMultifitParameters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  using InputProduct = EcalDigiDeviceCollection;
  using OutputProduct = EcalUncalibratedRecHitDeviceCollection;

  void launchKernels(Queue& queue,
                     InputProduct const& digisDevEB,
                     InputProduct const& digisDevEE,
                     OutputProduct& uncalibRecHitsDevEB,
                     OutputProduct& uncalibRecHitsDevEE,
                     EcalMultifitConditionsDevice const& conditionsDev,
                     EcalMultifitParameters const* paramsDev,
                     ConfigurationParameters const& configParams);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h
