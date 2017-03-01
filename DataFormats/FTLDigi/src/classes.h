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

  };
}

