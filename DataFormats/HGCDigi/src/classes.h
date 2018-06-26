#include <vector>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCDigi/interface/PHGCSimAccumulator.h"

namespace DataFormats_HGCDigi {
  struct dictionary {
    HGCSample anHGCsample;
    std::vector<HGCSample> vHGCsample;

    //Not specific
    HGCDataFrame<DetId,HGCSample> anDataFrame;
    std::vector<HGCDataFrame<DetId,HGCSample> > vDataFrames;
    edm::SortedCollection< HGCDataFrame<DetId,HGCSample> > scDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<DetId,HGCSample> > > prodDataFrames;

    // Sim cell accumulator (for premixing)
    PHGCSimAccumulator saHGC;
    PHGCSimAccumulator::Data saHGCdata;
    PHGCSimAccumulator::DetIdSize saHGCdis;
    std::vector<PHGCSimAccumulator::Data> vsaHGCdata;
    std::vector<PHGCSimAccumulator::DetIdSize> vsaHGCdis;
    edm::Wrapper<PHGCSimAccumulator> wsaHGC;
  };
}

