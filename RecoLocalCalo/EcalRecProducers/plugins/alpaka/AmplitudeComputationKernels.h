#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_AmplitudeComputationKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_AmplitudeComputationKernels_h

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DeclsForKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  using InputProduct = EcalDigiDeviceCollection;
  using OutputProduct = EcalUncalibratedRecHitDeviceCollection;

  void minimization_procedure(Queue& queue,
                              InputProduct const& digisDevEB,
                              InputProduct const& digisDevEE,
                              OutputProduct& uncalibRecHitsDevEB,
                              OutputProduct& uncalibRecHitsDevEE,
                              EventDataForScratchDevice& scratch,
                              EcalMultifitConditionsDevice const& conditionsDev,
                              ConfigurationParameters const& configParams,
                              uint32_t const totalChannels);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_AmplitudeComputationKernels_h
