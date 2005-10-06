#ifndef DIGIECAL_ECALDIGICOLLECTION_H
#define DIGIECAL_ECALDIGICOLLECTION_H

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "FWCore/EDProduct/interface/SortedCollection.h"

typedef edm::SortedCollection<EBDataFrame> EBDigiCollection;
typedef edm::SortedCollection<EEDataFrame> EEDigiCollection;
typedef edm::SortedCollection<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;

#endif
