#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<cms::EcalMGPASample> vMGPA_;
    std::vector<cms::EcalTriggerPrimitiveSample> vETPS_;

    std::vector<cms::EBDataFrame> vEB_;
    std::vector<cms::EEDataFrame> vEE_;
    std::vector<cms::EcalTriggerPrimitiveDigi> vETP_;

    cms::EBDigiCollection theEB_;
    cms::EEDigiCollection theEE_;
    cms::EcalTrigPrimDigiCollection theETP_;

    edm::Wrapper<cms::EBDigiCollection> theEBw_;
    edm::Wrapper<cms::EEDigiCollection> theEEw_;
    edm::Wrapper<cms::EcalTrigPrimDigiCollection> theETPw_; 
 }
}

