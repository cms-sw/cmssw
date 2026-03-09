#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthECLCCEpilogueArgsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthECLCCEpilogueArgsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthECLCCEpilogueArgsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFMultiDepthECLCCEpilogueArgsDeviceCollection = PortableCollection<::reco::PFMultiDepthECLCCEpilogueArgsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
