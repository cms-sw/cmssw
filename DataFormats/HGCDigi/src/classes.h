#include <vector>
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

    //HEX specific
    HGCDataFrame<HGCalDetId,HGCSample> anHGCalDataFrame;
    std::vector<HGCDataFrame<HGCalDetId,HGCSample> > vHGCalDataFrames;
    edm::SortedCollection< HGCDataFrame<HGCalDetId,HGCSample> > scHGCalDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<HGCalDetId,HGCSample> > > prodHGCalDataFrames;

    //BH (hcal) specific
    HGCDataFrame<HcalDetId,HGCSample> anHGCalBHDataFrame;
    std::vector<HGCDataFrame<HcalDetId,HGCSample> > vHGCalBHDataFrames;
    edm::SortedCollection< HGCDataFrame<HcalDetId,HGCSample> > scHGCalBHDataFrames;
    edm::Wrapper< edm::SortedCollection< HGCDataFrame<HcalDetId,HGCSample> > > prodHGCalBHDataFrames;

    // Sim cell accumulator (for premixing)
    PHGCSimAccumulator saHGC;
    PHGCSimAccumulator::Data saHGCdata;
    PHGCSimAccumulator::DetIdSize saHGCdis;
    std::vector<PHGCSimAccumulator::Data> vsaHGCdata;
    std::vector<PHGCSimAccumulator::DetIdSize> vsaHGCdis;
    edm::Wrapper<PHGCSimAccumulator> wsaHGC;
  };
}

