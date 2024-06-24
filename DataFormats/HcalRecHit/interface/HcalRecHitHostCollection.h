#ifndef DataFormats_HcalRecHit_HcalRecHitHostCollection_h
#define DataFormats_HcalRecHit_HcalRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"

namespace hcal {

  // HcalRecHitSoA in host memory
  using RecHitHostCollection = PortableHostCollection<HcalRecHitSoA>;
}  // namespace hcal

#endif
