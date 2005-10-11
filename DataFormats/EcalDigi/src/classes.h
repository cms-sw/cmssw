#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<EcalMGPASample> vMGPA_;
    std::vector<EcalFEMSample> vFEM_;
    std::vector<EcalTriggerPrimitiveSample> vETPS_;

    edm::SortedCollection<EBDataFrame> vEB_;
    edm::SortedCollection<EEDataFrame> vEE_;
    edm::SortedCollection<EcalTriggerPrimitiveDigi> vETP_;
    edm::SortedCollection<EcalPnDiodeDigi> vEPN_;

    EBDigiCollection theEB_;
    EEDigiCollection theEE_;
    EcalTrigPrimDigiCollection theETP_;
    EcalPnDiodeDigiCollection theEPN_;

    edm::Wrapper<EBDigiCollection> anotherEBw_;
    edm::Wrapper<EEDigiCollection> anotherEEw_;
    edm::Wrapper<EcalTrigPrimDigiCollection> anotherETPw_;
    edm::Wrapper<EcalPnDiodeDigiCollection> anotherEPNw_;

    edm::Wrapper< edm::SortedCollection<EBDataFrame> > theEBw_;
    edm::Wrapper< edm::SortedCollection<EEDataFrame> > theEEw_;
    edm::Wrapper< edm::SortedCollection<EcalTriggerPrimitiveDigi> > theETPw_; 
    edm::Wrapper< edm::SortedCollection<EcalPnDiodeDigi> > theEPNw_; 
 }
}

