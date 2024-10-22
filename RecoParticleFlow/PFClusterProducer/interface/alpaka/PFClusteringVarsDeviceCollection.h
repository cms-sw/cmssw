#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFClusteringVarsDevice_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFClusteringVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFClusteringVarsDeviceCollection = PortableCollection<::reco::PFClusteringVarsSoA>;
  // needs nRH allocation
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
