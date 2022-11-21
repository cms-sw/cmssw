#ifndef BeamConditionsMonitor_H
#define BeamConditionsMonitor_H

/** \class BeamConditionsMonitor
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *   
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

//
// class declaration
//
class BeamSpotObjectsRcd;
class BeamConditionsMonitor
    : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  BeamConditionsMonitor(const edm::ParameterSet&);
  ~BeamConditionsMonitor() override;

  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

protected:
  // BeginJob
  void beginJob() override;

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  // Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;

  // DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;

  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

  // Endjob
  void endJob() override;

private:
  edm::ParameterSet parameters_;
  std::string monitorName_;
  edm::InputTag bsSrc_;  // beam spot
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> beamSpotToken_;
  bool debug_;

  DQMStore* dbe_;

  int countEvt_;   //counter
  int countLumi_;  //counter

  // ----------member data ---------------------------
  BeamSpotObjects condBeamSpot;

  // MonitorElements
  MonitorElement* h_x0_lumi;
  MonitorElement* h_y0_lumi;
};

#endif
