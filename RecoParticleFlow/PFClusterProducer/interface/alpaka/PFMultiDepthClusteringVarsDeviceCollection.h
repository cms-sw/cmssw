#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringVarsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringVarsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFMultiDepthClusteringVarsDeviceCollection = PortableCollection<::reco::PFMultiDepthClusteringVarsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
