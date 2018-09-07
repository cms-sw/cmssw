#include <vector>
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

namespace DataFormats_FTLDigi {
  struct dictionary {
    FTLSample anFTLsample;
    std::vector<FTLSample> vFTLsample;

    FTLDataFrame anFTLDataFrame;
    std::vector<FTLDataFrame> vFTLDataFrames;
    edm::SortedCollection< FTLDataFrame > scFTLDataFrames;
    edm::Wrapper< edm::SortedCollection< FTLDataFrame > > prodFTlDataFrames;

    BTLSample aBTLsample;
    std::vector<BTLSample> vBTLsample;

    ETLSample anETLsample;
    std::vector<ETLSample> vETLsample;

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
