// -*- C++ -*-
//
// Package:    CalibTracker/SiStripDCS/plugins
// Class:      SyncDCSO2O
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

class FilterTrackerOn : public edm::EDFilter {
public:
  explicit FilterTrackerOn(const edm::ParameterSet&);
  ~FilterTrackerOn() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  int minModulesWithHVoff_;
  edm::ESGetToken<SiStripDetVOff, SiStripDetVOffRcd> detVOffToken_;
};
