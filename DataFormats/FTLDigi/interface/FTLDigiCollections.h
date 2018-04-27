#ifndef DIGIFTL_FTLDIGICOLLECTION_H
#define DIGIFTL_FTLDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/FTLDigi/interface/FTLDataFrameT.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLDigi/interface/FTLSample.h"

typedef FTLDataFrameT<BTLDetId,FTLSample> BTLDataFrame;
typedef edm::SortedCollection< BTLDataFrame > BTLDigiCollection;

typedef FTLDataFrameT<ETLDetId,FTLSample> ETLDataFrame;
typedef edm::SortedCollection< ETLDataFrame > ETLDigiCollection;


#endif
