// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripMonitorCondData
//
// Original Author:  Evelyne Delmeire
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"

class SiStripMonitorCondData : public edm::EDAnalyzer {
public:
  explicit SiStripMonitorCondData(const edm::ParameterSet &);

  ~SiStripMonitorCondData() override;

  void beginRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endRun(edm::Run const &run, edm::EventSetup const &eSetup) override;

private:
  SiStripClassToMonitorCondData moni_;
};

SiStripMonitorCondData::SiStripMonitorCondData(edm::ParameterSet const &iConfig)
    : moni_(iConfig, consumesCollector()) {}

SiStripMonitorCondData::~SiStripMonitorCondData() {}

void SiStripMonitorCondData::beginRun(edm::Run const &run, edm::EventSetup const &eSetup) {
  moni_.beginRun(run.run(), eSetup);
}

void SiStripMonitorCondData::analyze(edm::Event const &iEvent, edm::EventSetup const &eSetup) {
  moni_.analyseCondData(eSetup);
}

void SiStripMonitorCondData::endRun(edm::Run const &run, edm::EventSetup const &eSetup) {
  moni_.end();
  moni_.save();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripMonitorCondData);
