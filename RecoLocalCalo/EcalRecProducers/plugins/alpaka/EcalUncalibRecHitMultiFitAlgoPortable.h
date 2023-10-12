#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h

#include <vector>

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitParametersDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DeclsForKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  using InputProduct = EcalDigiDeviceCollection;
  using OutputProduct = EcalUncalibratedRecHitDeviceCollection;

  void launchKernels(Queue& queue,
                     InputProduct const& digisDevEB,
                     InputProduct const& digisDevEE,
                     OutputProduct& uncalibRecHitsDevEB,
                     OutputProduct& uncalibRecHitsDevEE,
                     EcalMultifitConditionsDevice const& conditionsDev,
                     EcalMultifitParametersDevice const& paramsDev,
                     ConfigurationParameters const& configParams);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalUncalibRecHitMultiFitAlgoPortable_h
