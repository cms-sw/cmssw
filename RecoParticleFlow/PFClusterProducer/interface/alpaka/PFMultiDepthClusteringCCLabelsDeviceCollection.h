#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringCCLabelsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthClusteringCCLabelsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringCCLabelsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFMultiDepthClusteringCCLabelsDeviceCollection = PortableCollection<::reco::PFMultiDepthClusteringCCLabelsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
