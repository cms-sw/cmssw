#include <vector>
#include "FWCore/EDProduct/interface/SortedCollection.h"
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
    
    edm::SortedCollection<HBHEDataFrame> vHBHE_;
    edm::SortedCollection<HODataFrame> vHO_;
    edm::SortedCollection<HFDataFrame> vHF_;
    edm::SortedCollection<HcalTriggerPrimitiveDigi> vHTP_;
    
    HBHEDigiCollection theHBHE_;
    HODigiCollection theHO_;
    HFDigiCollection theHF_;
    HcalTrigPrimDigiCollection theHTP_;

    edm::Wrapper<edm::SortedCollection<HBHEDataFrame> > anotherHBHE_;
    edm::Wrapper<edm::SortedCollection<HODataFrame> > anotherHO_;
    edm::Wrapper<edm::SortedCollection<HFDataFrame> > anotherHF_;
    edm::Wrapper<edm::SortedCollection<HcalTriggerPrimitiveDigi> > anotherHTP_;

    edm::Wrapper<HBHEDigiCollection> theHBHEw_;
    edm::Wrapper<HODigiCollection> theHOw_;
    edm::Wrapper<HFDigiCollection> theHFw_;
    edm::Wrapper<HcalTrigPrimDigiCollection> theHTPw_; 
 }
}

