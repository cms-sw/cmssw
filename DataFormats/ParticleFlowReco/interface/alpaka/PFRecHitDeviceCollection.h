#ifndef DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h
#define DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFRecHitHostCollection;
  using PFRecHitDeviceCollection = PortableCollection<::reco::PFRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::PFRecHitDeviceCollection, reco::PFRecHitHostCollection);

#endif  // DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h
