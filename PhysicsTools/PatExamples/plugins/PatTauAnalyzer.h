#ifndef PhysicsTools_PatExamples_PatTauAnalyzer_h  
#define PhysicsTools_PatExamples_PatTauAnalyzer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <TH1.h>

#include <string>

class PatTauAnalyzer : public edm::EDAnalyzer 
{
 public: 
  explicit PatTauAnalyzer(const edm::ParameterSet&);
  ~PatTauAnalyzer();
  
//--- methods inherited from EDAnalyzer base-class
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

 private:
//--- configuration parameters
  edm::InputTag src_;

  bool requireGenTauMatch_;

  std::string discrByLeadTrack_;
  std::string discrByIso_;
  std::string discrByTaNC_;

//--- generator level histograms
  TH1* hGenTauEnergy_;
  TH1* hGenTauPt_;
  TH1* hGenTauEta_;
  TH1* hGenTauPhi_;

//--- reconstruction level histograms
  TH1* hTauJetEnergy_;
  TH1* hTauJetPt_;
  TH1* hTauJetEta_;
  TH1* hTauJetPhi_;

  TH1* hNumTauJets_;
  
  TH1* hTauLeadTrackPt_;
  
  TH1* hTauNumSigConeTracks_;
  TH1* hTauNumIsoConeTracks_;

  TH1* hTauDiscrByIso_;
  TH1* hTauDiscrByTaNC_;
  TH1* hTauDiscrAgainstElectrons_;
  TH1* hTauDiscrAgainstMuons_;
  
  TH1* hTauJetEnergyIsoPassed_;
  TH1* hTauJetPtIsoPassed_;
  TH1* hTauJetEtaIsoPassed_;
  TH1* hTauJetPhiIsoPassed_;
  
  TH1* hTauJetEnergyTaNCpassed_;
  TH1* hTauJetPtTaNCpassed_;
  TH1* hTauJetEtaTaNCpassed_;
  TH1* hTauJetPhiTaNCpassed_;
};

#endif  


