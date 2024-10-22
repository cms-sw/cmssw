#ifndef DataFormats_ParticleFlowReco_interface_alpaka_PFClusterDeviceCollection_h
#define DataFormats_ParticleFlowReco_interface_alpaka_PFClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFClusterHostCollection;

  using PFClusterDeviceCollection = PortableCollection<::reco::PFClusterSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::PFClusterDeviceCollection, reco::PFClusterHostCollection);

#endif  // DataFormats_ParticleFlowReco_interface_alpaka_PFClusterDeviceCollection_h
