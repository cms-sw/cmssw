#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<HcalQIESample> vQIE_;
    std::vector<HcalTriggerPrimitiveSample> vTPS_;

    std::vector<HBHEDataFrame> vHBHE_;
    std::vector<HODataFrame> vHO_;
    std::vector<HFDataFrame> vHF_;
    std::vector<HcalTriggerPrimitiveDigi> vHTP_;

    HBHEDigiCollection theHBHE_;
    HODigiCollection theHO_;
    HFDigiCollection theHF_;
    HcalTrigPrimDigiCollection theHTP_;

    edm::Wrapper<HBHEDigiCollection> theHBHEw_;
    edm::Wrapper<HODigiCollection> theHOw_;
    edm::Wrapper<HFDigiCollection> theHFw_;
    edm::Wrapper<HcalTrigPrimDigiCollection> theHTPw_; 
 }
}

