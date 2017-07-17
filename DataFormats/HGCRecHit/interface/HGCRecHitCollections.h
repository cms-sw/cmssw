#ifndef DATAFORMATS_HGCRECHIT_HGCRECHITCOLLECTION_H
#define DATAFORMATS_HGCRECHIT_HGCRECHITCOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCUncalibratedRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


typedef edm::SortedCollection<HGCRecHit> HGCRecHitCollection;
typedef edm::Ref<HGCRecHitCollection> HGCRecHitRef;
typedef edm::RefVector<HGCRecHitCollection> HGCRecHitRefs;
typedef edm::RefProd<HGCRecHitCollection> HGCRecHitsRef;

typedef HGCRecHitCollection HGCeeRecHitCollection;
typedef HGCRecHitCollection HGChefRecHitCollection;
typedef HGCRecHitCollection HGChebRecHitCollection;

typedef edm::SortedCollection<HGCUncalibratedRecHit> HGCUncalibratedRecHitCollection;
typedef edm::Ref<HGCUncalibratedRecHitCollection> HGCUncalibratedRecHitRef;
typedef edm::RefVector<HGCUncalibratedRecHitCollection> HGCUncalibratedRecHitRefs;
typedef edm::RefProd<HGCUncalibratedRecHitCollection> HGCUncalibratedRecHitsRef;

typedef HGCUncalibratedRecHitCollection HGCeeUncalibratedRecHitCollection;
typedef HGCUncalibratedRecHitCollection HGChefUncalibratedRecHitCollection;
typedef HGCUncalibratedRecHitCollection HGChebUncalibratedRecHitCollection;

#endif
