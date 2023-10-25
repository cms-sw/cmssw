#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"

// This file defines two structs:
// 1) ALPAKA_ACCELERATOR_NAMESPACE::particleFlowRecHitProducer::HCAL
// 2) ALPAKA_ACCELERATOR_NAMESPACE::particleFlowRecHitProducer::ECAL
// These are used as template arguments of the PFRecHitSoAProducer class and
// related classes. This allows to specialise behaviour for the two calorimeter
// types.
namespace ALPAKA_ACCELERATOR_NAMESPACE::particleFlowRecHitProducer {

  struct HCAL {
    using CaloRecHitType = HBHERecHit;
    using CaloRecHitSoATypeHost = reco::CaloRecHitHostCollection;
    using CaloRecHitSoATypeDevice = reco::CaloRecHitDeviceCollection;
  };

  struct ECAL {
    using CaloRecHitType = EcalRecHit;
    using CaloRecHitSoATypeHost = reco::CaloRecHitHostCollection;
    using CaloRecHitSoATypeDevice = reco::CaloRecHitDeviceCollection;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::particleFlowRecHitProducer

#endif  // RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h
