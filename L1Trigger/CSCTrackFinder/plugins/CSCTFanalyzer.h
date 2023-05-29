#ifndef CSCTFAnalyzer_h
#define CSCTFAnalyzer_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

#include "TTree.h"
#include "TFile.h"

class CSCTFanalyzer : public edm::one::EDAnalyzer<> {
private:
  edm::InputTag lctProducer, mbProducer, dataTrackProducer, emulTrackProducer;
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> scalesToken;
  TTree* tree;
  TFile* file;
  int nDataMuons, nEmulMuons, verbose;
  double dphi1, deta1;
  int dpt1, dch1, dbx1;
  double dphi2, deta2;
  int dpt2, dch2, dbx2;
  double dphi3, deta3;
  int dpt3, dch3, dbx3;
  int drank1, drank2, drank3;
  int dmode1, dmode2, dmode3;
  int dlcts1, dlcts2, dlcts3;
  double ephi1, eeta1;
  int ept1, ech1, ebx1;
  double ephi2, eeta2;
  int ept2, ech2, ebx2;
  double ephi3, eeta3;
  int ept3, ech3, ebx3;
  int erank1, erank2, erank3;
  int emode1, emode2, emode3;

  const L1MuTriggerScales* ts;

public:
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;
  void endJob(void) override;
  void beginJob() override {}

  explicit CSCTFanalyzer(edm::ParameterSet const& pset);
  ~CSCTFanalyzer(void) override {}
};

#endif
