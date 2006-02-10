#ifndef DATAFORMATS_HCALRECHIT_HCALRECHITCOLLECTION_H
#define DATAFORMATS_HCALRECHIT_HCALRECHITCOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"

typedef edm::SortedCollection<HBHERecHit> HBHERecHitCollection;
typedef edm::SortedCollection<HORecHit> HORecHitCollection;
typedef edm::SortedCollection<HFRecHit> HFRecHitCollection;
typedef edm::SortedCollection<HcalTriggerPrimitiveRecHit> HcalTrigPrimRecHitCollection;

#endif
