// -*- C++ -*-
//
// Package:    HBHENoiseFilter
// Class:      HBHENoiseFilter
// 
/**\class HBHENoiseFilter

 Description: Filter that identifies events containing an RBX with bad pulse-shape, timing, hit multiplicity, and ADC 0 counts
              Designed to reduce noise rate by factor of 100

 Implementation:
              Use the HcalNoiseSummary to make cuts on an event-by-event basis
*/
//
// Original Author:  John Paul Chou (Brown/Rutgers)
//
//

#include <iostream>

#include "CommonTools/RecoAlgos/interface/HBHENoiseFilter.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HBHENoiseFilter::HBHENoiseFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  noiselabel_ = iConfig.getParameter<edm::InputTag>("noiselabel");
  minRatio_ = iConfig.getParameter<double>("minRatio");
  maxRatio_ = iConfig.getParameter<double>("maxRatio");
  minHPDHits_ = iConfig.getParameter<int>("minHPDHits");
  minRBXHits_ = iConfig.getParameter<int>("minRBXHits");
  minHPDNoOtherHits_ = iConfig.getParameter<int>("minHPDNoOtherHits");
  minZeros_ = iConfig.getParameter<int>("minZeros");
  minHighEHitTime_ = iConfig.getParameter<double>("minHighEHitTime");
  maxHighEHitTime_ = iConfig.getParameter<double>("maxHighEHitTime");
  maxRBXEMF_ = iConfig.getParameter<double>("maxRBXEMF");
  minNumIsolatedNoiseChannels_ = iConfig.getParameter<int>("minNumIsolatedNoiseChannels");
  minIsolatedNoiseSumE_ = iConfig.getParameter<double>("minIsolatedNoiseSumE");
  minIsolatedNoiseSumEt_ = iConfig.getParameter<double>("minIsolatedNoiseSumEt");  
  useTS4TS5_ = iConfig.getParameter<bool>("useTS4TS5");
}


HBHENoiseFilter::~HBHENoiseFilter()
{

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HBHENoiseFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  // get the Noise summary object
  edm::Handle<HcalNoiseSummary> summary_h;
  iEvent.getByLabel(noiselabel_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    return true;
  }
  const HcalNoiseSummary summary = *summary_h;

  //  if(summary.HasBadRBXTS4TS5() == true) std::cout << "TS4TS5 rejection!" << std::endl;
  //  else                                  std::cout << "TS4TS5 passing!" << std::endl;
  
  if(summary.minE2Over10TS()<minRatio_) return false;
  if(summary.maxE2Over10TS()>maxRatio_) return false;
  if(summary.maxHPDHits()>=minHPDHits_) return false;
  if(summary.maxRBXHits()>=minRBXHits_) return false;
  if(summary.maxHPDNoOtherHits()>=minHPDNoOtherHits_) return false;
  if(summary.maxZeros()>=minZeros_) return false;
  if(summary.min25GeVHitTime()<minHighEHitTime_) return false;
  if(summary.max25GeVHitTime()>maxHighEHitTime_) return false;
  if(summary.minRBXEMF()<maxRBXEMF_) return false;
  if(summary.numIsolatedNoiseChannels()>=minNumIsolatedNoiseChannels_) return false;
  if(summary.isolatedNoiseSumE()>=minIsolatedNoiseSumE_) return false;
  if(summary.isolatedNoiseSumEt()>=minIsolatedNoiseSumEt_) return false;
  if(useTS4TS5_ == true && summary.HasBadRBXTS4TS5() == true) return false;

  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HBHENoiseFilter);
