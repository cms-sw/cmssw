#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitParamsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitParamsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFRecHitECALParamsHostCollection;
  using ::reco::PFRecHitHCALParamsHostCollection;
  using PFRecHitECALParamsDeviceCollection = PortableCollection<::reco::PFRecHitECALParamsSoA>;
  using PFRecHitHCALParamsDeviceCollection = PortableCollection<::reco::PFRecHitHCALParamsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitParamsDeviceCollection_h
