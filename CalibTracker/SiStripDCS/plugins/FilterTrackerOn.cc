// -*- C++ -*-
//
// Package:    CalibTracker/SiStripDCS/plugins
// Class:      FilterTrackerOn
//
/**\class FilterTrackerOn FilterTrackerOn.cc

 Description: EDFilter returning true when the number of modules with HV on in the Tracker is
              above a given threshold.

*/
//
// Original Author:  Marco DE MATTIA
//         Created:  2010/03/10 10:51:00
//
//

// system include files
#include <memory>
#include <iostream>
#include <algorithm>

// user include files
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FilterTrackerOn : public edm::stream::EDFilter<> {
public:
  explicit FilterTrackerOn(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  ~FilterTrackerOn() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  int minModulesWithHVoff_;
  edm::ESGetToken<SiStripDetVOff, SiStripDetVOffRcd> detVOffToken_;
};

FilterTrackerOn::FilterTrackerOn(const edm::ParameterSet& iConfig)
    : minModulesWithHVoff_(iConfig.getParameter<int>("MinModulesWithHVoff")), detVOffToken_(esConsumes()) {}

FilterTrackerOn::~FilterTrackerOn() = default;

void FilterTrackerOn::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Filters out events in which at least a fraction of Tracker is DCS ON");
  desc.addUntracked<int>("MinModulesWithHVoff", 0);
  descriptions.addWithDefaultLabel(desc);
}

bool FilterTrackerOn::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  const auto& detVOff = iSetup.getData(detVOffToken_);

  LogDebug("FilterTrackerOn") << "detVOff.getHVoffCounts() = " << detVOff.getHVoffCounts() << " < "
                              << minModulesWithHVoff_;
  if (detVOff.getHVoffCounts() > minModulesWithHVoff_) {
    LogDebug("FilterTrackerOn") << " skipping event";
    return false;
  }
  LogDebug("FilterTrackerOn") << " keeping event";
  return true;
}

// EDFilter on the max number of modules with HV off
DEFINE_FWK_MODULE(FilterTrackerOn);
