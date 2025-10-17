#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringEdgeVarsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringEdgeVarsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringEdgeVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFMultiDepthClusteringEdgeVarsDeviceCollection = PortableCollection<::reco::PFMultiDepthClusteringEdgeVarsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
