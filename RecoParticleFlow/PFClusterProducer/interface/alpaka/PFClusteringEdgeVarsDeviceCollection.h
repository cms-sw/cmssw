#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFClusteringEdgeVarsDevice_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFClusteringEdgeVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringEdgeVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFClusteringEdgeVarsDeviceCollection = PortableCollection<::reco::PFClusteringEdgeVarsSoA>;
  // needs nRH + maxNeighbors allocation

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
