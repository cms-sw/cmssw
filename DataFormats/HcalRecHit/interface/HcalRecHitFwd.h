#ifndef DATAFORMATS_HCALRECHIT_HCALRECHITFWD_H
#define DATAFORMATS_HCALRECHIT_HCALRECHITFWD_H 1

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class HBHERecHit;
class HcalUpgradeRecHit;
class HORecHit;
class HFRecHit;
class ZDCRecHit;
class CastorRecHit;
class HcalCalibRecHit;

typedef edm::SortedCollection<HBHERecHit> HBHERecHitCollection;
typedef edm::Ref<HBHERecHitCollection> HBHERecHitRef;
typedef edm::RefVector<HBHERecHitCollection> HBHERecHitRefs;
typedef edm::RefProd<HBHERecHitCollection> HBHERecHitsRef;

typedef edm::SortedCollection<HcalUpgradeRecHit> HcalUpgradeRecHitCollection;
typedef edm::Ref<HcalUpgradeRecHitCollection> HcalUpgradeRecHitRef;
typedef edm::RefVector<HcalUpgradeRecHitCollection> HcalUpgradeRecHitRefs;
typedef edm::RefProd<HcalUpgradeRecHitCollection> HcalUpgradeRecHitsRef;

typedef edm::SortedCollection<HORecHit> HORecHitCollection;
typedef edm::Ref<HORecHitCollection> HORecHitRef;
typedef edm::RefVector<HORecHitCollection> HORecHitRefs;
typedef edm::RefProd<HORecHitCollection> HORecHitsRef;

typedef edm::SortedCollection<HFRecHit> HFRecHitCollection;
typedef edm::Ref<HFRecHitCollection> HFRecHitRef;
typedef edm::RefVector<HFRecHitCollection> HFRecHitRefs;
typedef edm::RefProd<HFRecHitCollection> HFRecHitsRef;

typedef edm::SortedCollection<ZDCRecHit> ZDCRecHitCollection;
typedef edm::Ref<ZDCRecHitCollection> ZDCRecHitRef;
typedef edm::RefVector<ZDCRecHitCollection> ZDCRecHitRefs;
typedef edm::RefProd<ZDCRecHitCollection> ZDCRecHitsRef;

typedef edm::SortedCollection<CastorRecHit> CastorRecHitCollection;
typedef edm::Ref<CastorRecHitCollection> CastorRecHitRef;
typedef edm::RefVector<CastorRecHitCollection> CastorRecHitRefs;
typedef edm::RefProd<CastorRecHitCollection> CastorRecHitsRef;

typedef edm::SortedCollection<HcalCalibRecHit> HcalCalibRecHitCollection;
typedef edm::Ref<HcalCalibRecHitCollection> HcalCalibRecHitRef;
typedef edm::RefVector<HcalCalibRecHitCollection> HcalCalibRecHitRefs;
typedef edm::RefProd<HcalCalibRecHitCollection> HcalCalibRecHitsRef;

#endif
