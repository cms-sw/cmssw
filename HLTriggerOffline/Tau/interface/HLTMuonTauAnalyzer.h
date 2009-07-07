#ifndef HLTriggerOffline_Muon_HLTMuonTauAnalyzer_H
#define HLTriggerOffline_Muon_HLTMuonTauAnalyzer_H

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTriggerOffline/Tau/interface/HLTMuonRate.h"
#include "TFile.h"
#include "TDirectory.h"

class HLTMuonTauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HLTMuonTauAnalyzer(const edm::ParameterSet&);
      ~HLTMuonTauAnalyzer();
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  int NumberOfTriggers;
  std::vector<HLTMuonRate *> muTriggerAnalyzer;
};

#endif
