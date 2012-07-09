#ifndef DATAFORMATS_HCALRECHIT_HCALRECHITCOLLECTION_H
#define DATAFORMATS_HCALRECHIT_HCALRECHITCOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalCalibRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalUpgradeRecHit.h"


typedef edm::SortedCollection<HBHERecHit> HBHERecHitCollection;
typedef edm::SortedCollection<HORecHit> HORecHitCollection;
typedef edm::SortedCollection<HFRecHit> HFRecHitCollection;
typedef edm::SortedCollection<ZDCRecHit> ZDCRecHitCollection;
typedef edm::SortedCollection<CastorRecHit> CastorRecHitCollection;
typedef edm::SortedCollection<HcalCalibRecHit> HcalCalibRecHitCollection;
typedef edm::SortedCollection<HcalUpgradeRecHit> HcalUpgradeRecHitCollection;

#endif
