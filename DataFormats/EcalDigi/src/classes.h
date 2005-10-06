#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<EcalMGPASample> vMGPA_;
    std::vector<EcalTriggerPrimitiveSample> vETPS_;

    edm::SortedCollection<EBDataFrame> vEB_;
    edm::SortedCollection<EEDataFrame> vEE_;
    edm::SortedCollection<EcalTriggerPrimitiveDigi> vETP_;

    EBDigiCollection theEB_;
    EEDigiCollection theEE_;
    EcalTrigPrimDigiCollection theETP_;

    edm::Wrapper<EBDigiCollection> anotherEBw_;
    edm::Wrapper<EEDigiCollection> anotherEEw_;
    edm::Wrapper<EcalTrigPrimDigiCollection> anotherETPw_;

    edm::Wrapper< edm::SortedCollection<EBDataFrame> > theEBw_;
    edm::Wrapper< edm::SortedCollection<EEDataFrame> > theEEw_;
    edm::Wrapper< edm::SortedCollection<EcalTriggerPrimitiveDigi> > theETPw_; 
 }
}

