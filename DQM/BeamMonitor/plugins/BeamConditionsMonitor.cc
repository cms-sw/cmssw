/*
 * \file BeamConditionsMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 *
 */

#include "DQM/BeamMonitor/plugins/BeamConditionsMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <numeric>
#include <cmath>
#include <TMath.h>
#include <iostream>
#include "TStyle.h"

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamConditionsMonitor::BeamConditionsMonitor(const ParameterSet& ps) : countEvt_(0), countLumi_(0) {
  parameters_ = ps;
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName", "YourSubsystemName");
  bsSrc_ = parameters_.getUntrackedParameter<InputTag>("beamSpot");
  debug_ = parameters_.getUntrackedParameter<bool>("Debug");
  beamSpotToken_ = esConsumes();
  dbe_ = Service<DQMStore>().operator->();

  if (!monitorName_.empty())
    monitorName_ = monitorName_ + "/";
}

BeamConditionsMonitor::~BeamConditionsMonitor() {}

//--------------------------------------------------------
void BeamConditionsMonitor::beginJob() {
  // book some histograms here
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_ + "Conditions");

  h_x0_lumi = dbe_->book1D("x0_lumi_cond", "x coordinate of beam spot vs lumi (Cond)", 10, 0, 10);
  h_x0_lumi->setAxisTitle("Lumisection", 1);
  h_x0_lumi->setAxisTitle("x_{0} (cm)", 2);
  h_x0_lumi->getTH1()->SetOption("E1");

  h_y0_lumi = dbe_->book1D("y0_lumi_cond", "y coordinate of beam spot vs lumi (Cond)", 10, 0, 10);
  h_y0_lumi->setAxisTitle("Lumisection", 1);
  h_y0_lumi->setAxisTitle("y_{0} (cm)", 2);
  h_y0_lumi->getTH1()->SetOption("E1");
}

//--------------------------------------------------------
void BeamConditionsMonitor::beginRun(const edm::Run& r, const EventSetup& context) {}

//--------------------------------------------------------
void BeamConditionsMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  countLumi_++;
}

// ----------------------------------------------------------
void BeamConditionsMonitor::analyze(const Event& iEvent, const EventSetup& iSetup) {
  countEvt_++;
  condBeamSpot = iSetup.getData(beamSpotToken_);
}

//--------------------------------------------------------
void BeamConditionsMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& iSetup) {
  LogInfo("BeamConditions") << "[BeamConditionsMonitor]:" << condBeamSpot << endl;
  h_x0_lumi->ShiftFillLast(condBeamSpot.GetX(), condBeamSpot.GetXError(), 1);
  h_y0_lumi->ShiftFillLast(condBeamSpot.GetY(), condBeamSpot.GetYError(), 1);
}
//--------------------------------------------------------
void BeamConditionsMonitor::endRun(const Run& r, const EventSetup& context) {}
//--------------------------------------------------------
void BeamConditionsMonitor::endJob() {}

DEFINE_FWK_MODULE(BeamConditionsMonitor);
