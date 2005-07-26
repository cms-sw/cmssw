#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<cms::HcalQIESample> vQIE_;
    std::vector<cms::HcalTriggerPrimitiveSample> vTPS_;

    std::vector<cms::HBHEDataFrame> vHBHE_;
    std::vector<cms::HODataFrame> vHO_;
    std::vector<cms::HFDataFrame> vHF_;
    std::vector<cms::HcalTriggerPrimitiveDigi> vHTP_;

    cms::HBHEDigiCollection theHBHE_;
    cms::HODigiCollection theHO_;
    cms::HFDigiCollection theHF_;
    cms::HcalTrigPrimDigiCollection theHTP_;

    edm::Wrapper<cms::HBHEDigiCollection> theHBHEw_;
    edm::Wrapper<cms::HODigiCollection> theHOw_;
    edm::Wrapper<cms::HFDigiCollection> theHFw_;
    edm::Wrapper<cms::HcalTrigPrimDigiCollection> theHTPw_; 
 }
}

