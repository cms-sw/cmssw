#ifndef DataFormats_HGCalDigi_HGCalDigiCollections_h
#define DataFormats_HGCalDigi_HGCalDigiCollections_h

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

typedef HGCROCChannelDataFrame<HGCalDetId> HGCROCChannelDataFrameSpec;
typedef edm::SortedCollection<HGCROCChannelDataFrameSpec> HGCalDigiCollection;

typedef HGCROCChannelDataFrame<HGCalElectronicsId> HGCROCChannelDataFrameElecSpec;
typedef edm::SortedCollection<HGCROCChannelDataFrameElecSpec> HGCalElecDigiCollection;

#endif
