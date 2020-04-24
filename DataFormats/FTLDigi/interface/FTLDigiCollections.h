#ifndef DIGIFTL_FTLDIGICOLLECTION_H
#define DIGIFTL_FTLDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/FTLDigi/interface/FTLDataFrameT.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "DataFormats/FTLDigi/interface/FTLSample.h"

typedef FTLDataFrameT<FastTimeDetId,FTLSample> FTLDataFrame;
typedef edm::SortedCollection< FTLDataFrame > FTLDigiCollection;

#endif
