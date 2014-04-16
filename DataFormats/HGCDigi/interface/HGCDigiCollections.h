#ifndef DIGIHGCAL_HGCALDIGICOLLECTION_H
#define DIGIHGCAL_HGCALDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HGCDataFrame.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"

typedef edm::SortedCollection< HGCDataFrame<HGCEEDetId,HGCSample> > HGCEEDigiCollection;
typedef edm::SortedCollection< HGCDataFrame<HGCHEDetId,HGCSample> > HGCHEDigiCollection;

#endif
