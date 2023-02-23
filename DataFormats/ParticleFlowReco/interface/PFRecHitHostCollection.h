#ifndef ParticleFlowReco_PFRecHitHostCollection_h
#define ParticleFlowReco_PFRecHitHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using PFRecHitHostCollection = PortableHostCollection<PFRecHitSoA>;

#endif