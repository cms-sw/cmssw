// -*- C++ -*-
//
// Package:    HLTEventNumberFilter
// Class:      HLTEventNumberFilter
//
/**\class HLTEventNumberFilter HLTEventNumberFilter.cc filter/HLTEventNumberFilter/src/HLTEventNumberFilter.cc

Description: 

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "HLTEventNumberFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constructors and destructor
//
HLTEventNumberFilter::HLTEventNumberFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  period_ = iConfig.getParameter<unsigned int>("period");
  invert_ = iConfig.getParameter<bool>("invert");
}

HLTEventNumberFilter::~HLTEventNumberFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HLTEventNumberFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("period", 4096);
  desc.add<bool>("invert", true);
  descriptions.add("hltEventNumberFilter", desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTEventNumberFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (iEvent.isRealData()) {
    bool accept(false);
    if (period_ != 0)
      accept = (((iEvent.id().event()) % period_) == 0);
    if (invert_)
      accept = !accept;
    return accept;
  } else {
    return true;
  }
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEventNumberFilter);
