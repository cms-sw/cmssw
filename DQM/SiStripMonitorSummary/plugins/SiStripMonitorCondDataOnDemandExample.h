#ifndef SiStripMonitorSummary_SiStripMonitorCondDataOnDemandExample_h
#define SiStripMonitorSummary_SiStripMonitorCondDataOnDemandExample_h
// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripMonitorCondDataOnDemandExample
//
// Original Author:  Evelyne Delmeire
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <string>
#include <vector>

class SiStripClassToMonitorCondData;

class SiStripMonitorCondDataOnDemandExample : public edm::EDAnalyzer {
public:
  explicit SiStripMonitorCondDataOnDemandExample(const edm::ParameterSet &);

  ~SiStripMonitorCondDataOnDemandExample() override;

  void beginJob() override;
  void beginRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void endJob() override;

private:
  int eventCounter_;
  edm::ParameterSet conf_;
  SiStripClassToMonitorCondData *condDataMonitoring_;
};

#endif
