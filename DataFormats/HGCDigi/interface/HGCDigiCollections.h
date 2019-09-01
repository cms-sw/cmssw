#ifndef DIGIHGCAL_HGCALDIGICOLLECTION_H
#define DIGIHGCAL_HGCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"

typedef HGCDataFrame<HGCalDetId, HGCSample> HGCEEDataFrame;
typedef edm::SortedCollection<HGCEEDataFrame> HGCEEDigiCollection;

typedef HGCDataFrame<HGCalDetId, HGCSample> HGCHEDataFrame;
typedef edm::SortedCollection<HGCHEDataFrame> HGCHEDigiCollection;

typedef HGCDataFrame<HcalDetId, HGCSample> HGCBHDataFrame;
typedef edm::SortedCollection<HGCBHDataFrame> HGCBHDigiCollection;

typedef HGCDataFrame<DetId, HGCSample> HGCalDataFrame;
typedef edm::SortedCollection<HGCalDataFrame> HGCalDigiCollection;

#endif
