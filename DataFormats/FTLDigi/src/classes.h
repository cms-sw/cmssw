#include <vector>
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

namespace DataFormats_FTLDigi {
  struct dictionary {

    FTLSample anFTLsample;
    std::vector<FTLSample> vFTLsample;

    BTLDataFrame anBTLDataFrame;
    std::vector<BTLDataFrame> vBTLDataFrames;
    edm::SortedCollection< BTLDataFrame > scBTLDataFrames;
    edm::Wrapper< edm::SortedCollection< BTLDataFrame > > prodBTLDataFrames;

    ETLDataFrame anETLDataFrame;
    std::vector<ETLDataFrame> vETLDataFrames;
    edm::SortedCollection< ETLDataFrame > scETLDataFrames;
    edm::Wrapper< edm::SortedCollection< ETLDataFrame > > prodETLDataFrames;

  };
}

