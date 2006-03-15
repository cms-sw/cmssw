#ifndef RAWECAL_ECALRAWCOLLECTION_H
#define RAWECAL_ECALRAWCOLLECTION_H

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/Common/interface/SortedCollection.h"

typedef edm::SortedCollection<EcalDCCHeaderBlock> EcalRawDataCollection;

#endif
