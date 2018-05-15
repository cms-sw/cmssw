#ifndef DIGIFTL_FTLDIGICOLLECTION_H
#define DIGIFTL_FTLDIGICOLLECTION_H

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/FTLDigi/interface/FTLDataFrameT.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "DataFormats/FTLDigi/interface/FTLSample.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLDigi/interface/BTLSample.h"
#include "DataFormats/FTLDigi/interface/ETLSample.h"

typedef FTLDataFrameT<FastTimeDetId,FTLSample> FTLDataFrame;
typedef edm::SortedCollection< FTLDataFrame > FTLDigiCollection;

typedef FTLDataFrameT<BTLDetId,BTLSample> BTLDataFrame;
typedef edm::SortedCollection< BTLDataFrame > BTLDigiCollection;

typedef FTLDataFrameT<ETLDetId,ETLSample> ETLDataFrame;
typedef edm::SortedCollection< ETLDataFrame > ETLDigiCollection;

#endif
