#include "CalibTracker/SiStripDCS/plugins/FilterTrackerOn.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <algorithm>

FilterTrackerOn::FilterTrackerOn(const edm::ParameterSet& iConfig)
    : minModulesWithHVoff_(iConfig.getParameter<int>("MinModulesWithHVoff")), detVOffToken_(esConsumes()) {}

FilterTrackerOn::~FilterTrackerOn() {}

bool FilterTrackerOn::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto& detVOff = iSetup.getData(detVOffToken_);

  // std::cout << "detVOff.getHVoffCounts() = " << detVOff.getHVoffCounts() << " < " << minModulesWithHVoff_;
  if (detVOff.getHVoffCounts() > minModulesWithHVoff_) {
    // std::cout << " skipping event" << std::endl;
    return false;
  }
  // cout << " keeping event" << endl;
  return true;
}

// ------------ method called once each job just before starting event loop  ------------
void FilterTrackerOn::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void FilterTrackerOn::endJob() {}
