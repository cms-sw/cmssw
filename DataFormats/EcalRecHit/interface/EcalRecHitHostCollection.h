#ifndef DataFormats_EcalRecHit_EcalRecHitHostCollection_h
#define DataFormats_EcalRecHit_EcalRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitSoA.h"

// EcalRecHitSoA in host memory
using EcalRecHitHostCollection = PortableHostCollection<EcalRecHitSoA>;

#endif
