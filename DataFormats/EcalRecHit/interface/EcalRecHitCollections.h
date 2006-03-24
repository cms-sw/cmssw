#ifndef DATAFORMATS_ECALRECHIT_ECALRECHITCOLLECTION_H
#define DATAFORMATS_ECALRECHIT_ECALRECHITCOLLECTION_H

#include "FWCore/EDProduct/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"


typedef edm::SortedCollection<EcalRecHit> EcalRecHitCollection;
typedef edm::SortedCollection<EcalUncalibratedRecHit> EcalUncalibratedRecHitCollection;


#endif
