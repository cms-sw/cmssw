#ifndef DATAFORMATS_FTLRECHIT_FTLRECHITCOLLECTION_H
#define DATAFORMATS_FTLRECHIT_FTLRECHITCOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


typedef edm::SortedCollection<FTLRecHit> FTLRecHitCollection;
typedef edm::Ref<FTLRecHitCollection> FTLRecHitRef;
typedef edm::RefVector<FTLRecHitCollection> FTLRecHitRefs;
typedef edm::RefProd<FTLRecHitCollection> FTLRecHitsRef;

typedef edm::SortedCollection<FTLUncalibratedRecHit> FTLUncalibratedRecHitCollection;
typedef edm::Ref<FTLUncalibratedRecHitCollection> FTLUncalibratedRecHitRef;
typedef edm::RefVector<FTLUncalibratedRecHitCollection> FTLUncalibratedRecHitRefs;
typedef edm::RefProd<FTLUncalibratedRecHitCollection> FTLUncalibratedRecHitsRef;


#endif
