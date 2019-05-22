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

namespace mtdhelpers {
  struct FTLRowColDecode {
    static inline int row(const DetId& id, const std::vector<FTLSample>& data) { return -1; } // no rows or columns
    static inline int col(const DetId& id, const std::vector<FTLSample>& data) { return -1; }
  };

  struct BTLRowColDecode {
    static inline int row(const DetId& id, const std::vector<BTLSample>& data) { return data.front().row(); }
    static inline int col(const DetId& id, const std::vector<BTLSample>& data) { return data.front().column(); }
  };

  struct ETLRowColDecode {
    static inline int row(const DetId& id, const std::vector<ETLSample>& data) { return data.front().row(); }
    static inline int col(const DetId& id, const std::vector<ETLSample>& data) { return data.front().column(); }
  };
}

typedef FTLDataFrameT<FastTimeDetId,FTLSample,mtdhelpers::FTLRowColDecode> FTLDataFrame;
typedef edm::SortedCollection< FTLDataFrame > FTLDigiCollection;

typedef FTLDataFrameT<BTLDetId,BTLSample,mtdhelpers::BTLRowColDecode> BTLDataFrame;
typedef edm::SortedCollection< BTLDataFrame > BTLDigiCollection;

typedef FTLDataFrameT<ETLDetId,ETLSample,mtdhelpers::ETLRowColDecode> ETLDataFrame;
typedef edm::SortedCollection< ETLDataFrame > ETLDigiCollection;

#endif
