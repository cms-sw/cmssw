#ifndef ParticleFlowReco_CaloRecHitHostCollection_h
#define ParticleFlowReco_CaloRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"

using CaloRecHitHostCollection = PortableHostCollection<CaloRecHitSoA>;

#endif