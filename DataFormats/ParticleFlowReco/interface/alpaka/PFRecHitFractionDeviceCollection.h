#ifndef DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitFractionDeviceCollection_h
#define DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitFractionDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFRecHitFractionHostCollection;

  using PFRecHitFractionDeviceCollection = PortableCollection<::reco::PFRecHitFractionSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::PFRecHitFractionDeviceCollection, reco::PFRecHitFractionHostCollection);

#endif  // DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitFractionDeviceCollection_h
