#ifndef DIGIHGCAL_HGCALDIGICOLLECTION_H
#define DIGIHGCAL_HGCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"

typedef HGCDataFrame<HGCEEDetId,HGCSample>      HGCEEDataFrame;
typedef edm::SortedCollection< HGCEEDataFrame > HGCEEDigiCollection;

typedef HGCDataFrame<HGCHEDetId,HGCSample>           HGCHEDataFrame;
typedef edm::SortedCollection< HGCHEDataFrame > HGCHEDigiCollection;


#endif
