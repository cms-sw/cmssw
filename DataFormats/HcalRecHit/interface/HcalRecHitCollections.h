#ifndef RECHITHCAL_HCALRECHITCOLLECTION_H
#define RECHITHCAL_HCALRECHITCOLLECTION_H

#include <vector>
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"

namespace cms {

  typedef std::vector<HBHERecHit> HBHERecHitCollection;
  typedef std::vector<HORecHit> HORecHitCollection;
  typedef std::vector<HFRecHit> HFRecHitCollection;
  typedef std::vector<HcalTriggerPrimitiveRecHit> HcalTrigPrimRecHitCollection;

}

#endif
