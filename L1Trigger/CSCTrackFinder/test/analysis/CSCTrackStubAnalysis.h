#ifndef CSCTrackFinder_CSCTrackStubAnalysis_h
#define CSCTrackFinder_CSCTrackStubAnalysis_h

/**
 * \author L. Gray 6/17/06
 *
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//ROOT
#include <TH1F.h>
#include <TH1D.h>
#include <TH1I.h>
#include <TH2I.h>

class CSCTrackStubAnalysis : public edm::EDAnalyzer {
 public:
  explicit CSCTrackStubAnalysis(edm::ParameterSet const& conf);
  virtual ~CSCTrackStubAnalysis() {}
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void beginJob();
 private:
  // variables persistent across events should be declared here.
  //

  void DeleteHistos();

  std::pair<int,int> least, greatest;

  Int_t cntdtts, cntcscts;

  // DT related histos
  TH1I *hDTts_phi;

  //CSC related histos
  TH1I *hCSCts_phi;

  // DT - CSC Comparison
  TH2I *hDTvsCSC_phi, *hDTvsCSC_phi_corr;



};

DEFINE_FWK_MODULE(CSCTrackStubAnalysis);

#endif
