#ifndef DATAFORMATS_ECALRECHIT_ECALRECHITCOLLECTION_H
#define DATAFORMATS_ECALRECHIT_ECALRECHITCOLLECTION_H

#include "FWCore/EDProduct/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

typedef edm::SortedCollection<EcalRecHit> EcalRecHitCollection;
typedef edm::Ref<EcalRecHitCollection> EcalRecHitRef;
typedef edm::RefVector<EcalRecHitCollection> EcalRecHitRefs;
typedef edm::RefProd<EcalRecHitCollection> EcalRecHitsRef;

typedef edm::SortedCollection<EcalUncalibratedRecHit> EcalUncalibratedRecHitCollection;
typedef edm::Ref<EcalUncalibratedRecHitCollection> EcalUncalibratedRecHitRef;
typedef edm::RefVector<EcalUncalibratedRecHitCollection> EcalUncalibratedRecHitRefs;
typedef edm::RefProd<EcalUncalibratedRecHitCollection> EcalUncalibratedRecHitsRef;

#endif
