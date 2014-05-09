#include <vector>
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

namespace {
  struct dictionary {

    HGCSample anHGCsample;
    HGCEEDetId anHGCEEDetId;
    HGCDataFrame<HGCEEDetId,HGCSample> anHGCEEDataFrame;
    std::vector<HGCDataFrame<HGCEEDetId,HGCSample> > vHGCEEDataFrames;
    edm::SortedCollection< HGCDataFrame<HGCEEDetId,HGCSample> > scHGCEEDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<HGCEEDetId,HGCSample> > > prodHGCEEDataFrames;
    HGCEEDigiCollection theHGCEE_;

  };
}

