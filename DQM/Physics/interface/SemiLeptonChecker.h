#ifndef DQM_SemiLepChecker_h
#define DQM_SemiLepChecker_h

// system include files
#include <memory>
#include <string>
#include <vector>
// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"

//MuonPlusJets
#include "DQM/Physics/interface/JetCombinatorics.h"

//jet corrections
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/**
   \class   SemiLeptonChecker SemiLeptonChecker.h "DQM/Physics/interface/SemiLeptonChecker.h"

   \brief   Add a one sentence description here...

   Module dedicated to all objects: 
   Electrons, Muons, CaloJets, CaloMets
   It takes a vector as input instead of edm::View or edm::Handle
   -> possibility to have a selected collection
   It's based on EDAnalyzer
   It uses DQMStore to store the histograms
   in this directories: 
   relativePath_+"Muons_"+label_
   relativePath_+"Electrons_"+label_
   relativePath_+"CaloJets_"+label_
   relativePath_+"CaloMETs_"+label_
   This module has to be called by a channel-specific module like LeptonJetsChecker
*/

class SemiLeptonChecker{
 public:
  explicit SemiLeptonChecker(const edm::ParameterSet&, std::string relativePath, std::string label);
  ~SemiLeptonChecker();
  
  
  void beginJob(const edm::EventSetup& iSetup, std::string jetCorrector) ;
  void analyze(const std::vector<reco::CaloJet>& jets, bool useJES, const std::vector<reco::CaloMET>& mets, const std::vector<reco::Muon>& muons, const std::vector<reco::GsfElectron>& electrons, const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void endJob() ;
  
  bool goodMET() {return found_goodMET_;}
 private:
  
  std::string relativePath_; //use for the name of the directory
  std::string label_; //use for the name of the directory
  int NbOfEvents;
  
  //Histograms are booked in the beginJob() method
  std::map<std::string,MonitorElement*> histocontainerC_;
  std::map<std::string,MonitorElement*> histocontainer_[2];
  
  DQMStore* dqmStore_;
  
  //Jet corrections
  const JetCorrector *acorrector ;
  
  JetCombinatorics myCombi0_;
  JetCombinatorics myCombi1_;
  bool found_goodMET_;
  bool isMuon_;
  std::string leptonType_;
};

#endif
