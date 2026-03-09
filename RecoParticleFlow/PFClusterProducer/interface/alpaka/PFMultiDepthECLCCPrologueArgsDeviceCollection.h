#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthECLCCPrologueArgsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFMultiDepthECLCCPrologueArgsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthECLCCPrologueArgsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFMultiDepthECLCCPrologueArgsDeviceCollection = PortableCollection<::reco::PFMultiDepthECLCCPrologueArgsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
