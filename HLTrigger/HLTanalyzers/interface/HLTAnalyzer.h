#include <iostream>

#include "HLTrigger/HLTAnalyzers/interface/HLTEgamma.h"
#include "HLTrigger/HLTAnalyzers/interface/HLTInfo.h"
#include "HLTrigger/HLTAnalyzers/interface/HLTJets.h"
#include "HLTrigger/HLTAnalyzers/interface/HLTMCtruth.h"
#include "HLTrigger/HLTAnalyzers/interface/HLTMuon.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

/** \class HLTAnalyzer
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */

class HLTAnalyzer : public edm::EDAnalyzer {
public:
  explicit HLTAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();

  // Analysis tree to be filled
  TTree *HltTree;

private:
  // variables persistent across events should be declared here.
  //
  ///Default analyses
  HLTEgamma elm_analysis_;
  HLTJets jet_analysis_;
  HLTMCtruth mct_analysis_;
  HLTMuon muon_analysis_;
  HLTInfo hlt_analysis_;

  std::string recjets_,genjets_,recmet_,genmet_,calotowers_,hltobj_,hltresults_;
  std::string pixElectron_,silElectron_,Photon_,muon_;
  std::string l1extramc_; 
  int errCnt;
  const int errMax(){return 100;}

  string _HistName; // Name of histogram file
  double _EtaMin,_EtaMax;
  TFile* m_file; // pointer to Histogram file

};
