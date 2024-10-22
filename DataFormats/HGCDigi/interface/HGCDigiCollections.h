#ifndef DIGIHGCAL_HGCALDIGICOLLECTION_H
#define DIGIHGCAL_HGCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"

typedef HGCDataFrame<DetId, HGCSample> HGCalDataFrame;
typedef edm::SortedCollection<HGCalDataFrame> HGCalDigiCollection;

#endif
