#ifndef RAWECAL_ECALRAWCOLLECTION_H
#define RAWECAL_ECALRAWCOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

typedef edm::SortedCollection<EcalDCCHeaderBlock> EcalRawDataCollection;
typedef edm::SortedCollection<ESDCCHeaderBlock> ESRawDataCollection;
typedef edm::SortedCollection<ESKCHIPBlock> ESLocalRawDataCollection;

#endif
