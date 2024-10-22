#ifndef RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsHostCollection_h
#define RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterParamsSoA.h"

namespace reco {

  using PFClusterParamsHostCollection = PortableHostCollection<PFClusterParamsSoA>;

}

#endif
