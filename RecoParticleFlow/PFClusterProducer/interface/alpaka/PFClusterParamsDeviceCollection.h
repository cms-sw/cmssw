#ifndef RecoParticleFlow_PFClusterProducer_interface_alpaka_PFClusterParamsDeviceCollection_h
#define RecoParticleFlow_PFClusterProducer_interface_alpaka_PFClusterParamsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterParamsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterParamsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFClusterParamsHostCollection;

  using PFClusterParamsDeviceCollection = PortableCollection<::reco::PFClusterParamsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

// check that the portable device collection for the host device is the same as the portable host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::PFClusterParamsDeviceCollection, reco::PFClusterParamsHostCollection);

#endif
