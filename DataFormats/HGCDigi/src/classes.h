#include <vector>
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCDigi/interface/PHGCSimAccumulator.h"

namespace DataFormats_HGCDigi {
  struct dictionary {
    HGCSample anHGCsample;
    std::vector<HGCSample> vHGCsample;

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

    // Sim cell accumulator (for premixing)
    PHGCSimAccumulator saHGC;
    PHGCSimAccumulator::Data saHGCdata;
    PHGCSimAccumulator::DetIdSize saHGCdis;
    std::vector<PHGCSimAccumulator::Data> vsaHGCdata;
    std::vector<PHGCSimAccumulator::DetIdSize> vsaHGCdis;
    edm::Wrapper<PHGCSimAccumulator> wsaHGC;
  };
}

