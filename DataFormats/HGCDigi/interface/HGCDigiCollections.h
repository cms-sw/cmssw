#ifndef DIGIHGCAL_HGCALDIGICOLLECTION_H
#define DIGIHGCAL_HGCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"

typedef HGCDataFrame<DetId,HGCSample>           HGCEEDataFrame;
typedef edm::SortedCollection< HGCEEDataFrame > HGCEEDigiCollection;

typedef HGCDataFrame<DetId,HGCSample>           HGCHEDataFrame;
typedef edm::SortedCollection< HGCHEDataFrame > HGCHEDigiCollection;

typedef HGCDataFrame<DetId,HGCSample>           HGCBHDataFrame;
typedef edm::SortedCollection< HGCBHDataFrame > HGCBHDigiCollection;

#endif
