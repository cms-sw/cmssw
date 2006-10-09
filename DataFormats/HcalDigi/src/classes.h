#include <vector>
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<HcalQIESample> vQIE_;
    std::vector<HcalTriggerPrimitiveSample> vTPS_;
    
    edm::SortedCollection<HBHEDataFrame> vHBHE_;
    edm::SortedCollection<HODataFrame> vHO_;
    edm::SortedCollection<HFDataFrame> vHF_;
    edm::SortedCollection<HcalCalibDataFrame> vHC_;
    edm::SortedCollection<HcalTriggerPrimitiveDigi> vHTP_;
    edm::SortedCollection<HcalHistogramDigi> vHH_;

    HBHEDigiCollection theHBHE_;
    HODigiCollection theHO_;
    HFDigiCollection theHF_;
    HcalCalibDigiCollection theHC_;
    HcalTrigPrimDigiCollection theHTP_;
    HcalHistogramDigiCollection theHH_;
    ZDCDigiCollection theZDC_;

    edm::Wrapper<edm::SortedCollection<HBHEDataFrame> > anotherHBHE_;
    edm::Wrapper<edm::SortedCollection<HODataFrame> > anotherHO_;
    edm::Wrapper<edm::SortedCollection<HFDataFrame> > anotherHF_;
    edm::Wrapper<edm::SortedCollection<HcalCalibDataFrame> > anotherHC_;
    edm::Wrapper<edm::SortedCollection<HcalTriggerPrimitiveDigi> > anotherHTP_;
    edm::Wrapper<edm::SortedCollection<HcalHistogramDigi> > anotherHH_;
    edm::Wrapper<edm::SortedCollection<ZDCDataFrame> > anotherZDC_;

    edm::Wrapper<HBHEDigiCollection> theHBHEw_;
    edm::Wrapper<HODigiCollection> theHOw_;
    edm::Wrapper<HFDigiCollection> theHFw_;
    edm::Wrapper<HcalCalibDigiCollection> theHCw_;
    edm::Wrapper<HcalTrigPrimDigiCollection> theHTPw_; 
    edm::Wrapper<HcalHistogramDigiCollection> theHHw_; 
    edm::Wrapper<HcalUnpackerReport> theReport_;
 }
}

