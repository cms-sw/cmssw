#include <vector>
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

namespace DataFormats_HGCDigi {
  struct dictionary {
    HGCSample anHGCsample;
    std::vector<HGCSample> vHGCsample;

    //EE specific
    HGCDataFrame<HGCEEDetId,HGCSample> anHGCEEDataFrame;
    std::vector<HGCDataFrame<HGCEEDetId,HGCSample> > vHGCEEDataFrames;
    edm::SortedCollection< HGCDataFrame<HGCEEDetId,HGCSample> > scHGCEEDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<HGCEEDetId,HGCSample> > > prodHGCEEDataFrames;
    HGCEEDigiCollection dcHGCEE;
    edm::Wrapper<HGCEEDigiCollection> wdcHGCEE;

    //HE specific
    HGCDataFrame<HGCHEDetId,HGCSample> anHGCHEDataFrame;
    std::vector<HGCDataFrame<HGCHEDetId,HGCSample> > vHGCHEDataFrames;
    edm::SortedCollection< HGCDataFrame<HGCHEDetId,HGCSample> > scHGCHEDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<HGCHEDetId,HGCSample> > > prodHGCHEDataFrames;
    HGCHEDigiCollection dcHGCHE;
    edm::Wrapper<HGCHEDigiCollection> wdcHGCHE;
  };
}

