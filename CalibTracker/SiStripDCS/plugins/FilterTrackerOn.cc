#include "CalibTracker/SiStripDCS/plugins/FilterTrackerOn.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <algorithm>
#include <iostream>

FilterTrackerOn::FilterTrackerOn(const edm::ParameterSet &iConfig)
    : minModulesWithHVoff_(iConfig.getParameter<int>("MinModulesWithHVoff")) {}

FilterTrackerOn::~FilterTrackerOn() {}

bool FilterTrackerOn::filter(edm::Event &iEvent,
                             const edm::EventSetup &iSetup) {
  using namespace edm;

  ESHandle<SiStripDetVOff> detVOff;
  iSetup.get<SiStripDetVOffRcd>().get(detVOff);

  // std::cout << "detVOff->getHVoffCounts() = " << detVOff->getHVoffCounts() <<
  // " < " << minModulesWithHVoff_;
  if (detVOff->getHVoffCounts() > minModulesWithHVoff_) {
    // std::cout << " skipping event" << std::endl;
    return false;
  }
  // cout << " keeping event" << endl;
  return true;
}

// ------------ method called once each job just before starting event loop
// ------------
void FilterTrackerOn::beginJob() {}

// ------------ method called once each job just after ending the event loop
// ------------
void FilterTrackerOn::endJob() {}
