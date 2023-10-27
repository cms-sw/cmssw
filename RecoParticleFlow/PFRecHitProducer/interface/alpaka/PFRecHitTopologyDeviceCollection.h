#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitTopologyDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitTopologyDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFRecHitECALTopologyHostCollection;
  using ::reco::PFRecHitHCALTopologyHostCollection;
  using PFRecHitHCALTopologyDeviceCollection = PortableCollection<::reco::PFRecHitHCALTopologySoA>;
  using PFRecHitECALTopologyDeviceCollection = PortableCollection<::reco::PFRecHitECALTopologySoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitTopologyDeviceCollection_h
