#ifndef DataFormats_EcalRecHit_EcalUncalibratedRecHitHostCollection_h
#define DataFormats_EcalRecHit_EcalUncalibratedRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"

// EcalUncalibratedRecHitSoA in host memory
using EcalUncalibratedRecHitHostCollection = PortableHostCollection<EcalUncalibratedRecHitSoA>;

#endif
