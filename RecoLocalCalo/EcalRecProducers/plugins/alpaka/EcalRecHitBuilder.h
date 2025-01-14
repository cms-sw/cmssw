#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalRecHitBuilder_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalRecHitBuilder_h

//
// Builder of ECAL RecHits on GPU
//

#include "CondFormats/EcalObjects/interface/EcalRecHitParameters.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DeclsForKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit {

  using InputProduct = EcalUncalibratedRecHitDeviceCollection;
  using OutputProduct = EcalRecHitDeviceCollection;

  // host version, to be called by the plugin
  void create_ecal_rechit(Queue& queue,
                          InputProduct const* ebUncalibRecHits,
                          InputProduct const* eeUncalibRecHits,
                          OutputProduct& ebRecHits,
                          OutputProduct& eeRecHits,
                          EcalRecHitConditionsDevice const& conditionsDev,
                          EcalRecHitParameters const* parametersDev,
                          edm::TimeValue_t const& eventTime,
                          ConfigurationParameters const& configParams,
                          bool const isPhase2);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_EcalRecHitBuilder_h
