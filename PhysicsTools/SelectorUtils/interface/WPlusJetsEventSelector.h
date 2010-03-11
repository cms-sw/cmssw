#ifndef PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h
#define PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h

#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "PhysicsTools/SelectorUtils/interface/ElectronVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/MuonVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

class WPlusJetsEventSelector : public EventSelector {
 public:
  WPlusJetsEventSelector( edm::ParameterSet const & params );

  virtual void scaleJets(double scale) {jetScale_ = scale;}
  
  virtual bool operator()( edm::EventBase const & t, std::strbitset & ret);
  using EventSelector::operator();

  std::vector<reco::ShallowClonePtrCandidate> const & selectedJets     () const { return selectedJets_;     } 
  std::vector<reco::ShallowClonePtrCandidate> const & cleanedJets      () const { return cleanedJets_;      } 
  std::vector<reco::ShallowClonePtrCandidate> const & selectedElectrons() const { return selectedElectrons_;}
  std::vector<reco::ShallowClonePtrCandidate> const & selectedMuons    () const { return selectedMuons_;    }
 
 protected: 

  edm::InputTag               muonTag_;
  edm::InputTag               electronTag_;
  edm::InputTag               jetTag_;
  edm::InputTag               metTag_;
  edm::InputTag               trigTag_;

  std::vector<reco::ShallowClonePtrCandidate> selectedJets_;
  std::vector<reco::ShallowClonePtrCandidate> selectedMuons_;
  std::vector<reco::ShallowClonePtrCandidate> selectedElectrons_;
  std::vector<reco::ShallowClonePtrCandidate> looseMuons_;
  std::vector<reco::ShallowClonePtrCandidate> looseElectrons_;
  std::vector<reco::ShallowClonePtrCandidate> selectedMETs_;
  std::vector<reco::ShallowClonePtrCandidate> cleanedJets_;
  std::vector<reco::ShallowClonePtrCandidate> selectedElectrons2_;

  MuonVPlusJetsIDSelectionFunctor      muonIdTight_;
  ElectronVPlusJetsIDSelectionFunctor  electronIdTight_;
  MuonVPlusJetsIDSelectionFunctor      muonIdLoose_;
  ElectronVPlusJetsIDSelectionFunctor  electronIdLoose_;
  JetIDSelectionFunctor                jetIdLoose_;

  int minJets_;
  double dR_;
  bool muPlusJets_;
  bool ePlusJets_;

  double muPtMin_  ;
  double muEtaMax_ ;
  double elePtMin_ ;
  double eleEtaMax_;

  double muPtMinLoose_  ;
  double muEtaMaxLoose_ ;
  double elePtMinLoose_ ;
  double eleEtaMaxLoose_;

  double jetPtMin_ ;
  double jetEtaMax_;

  double jetScale_;

  
};


#endif
