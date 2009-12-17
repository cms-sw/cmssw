#ifndef PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h
#define PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h

#include "PhysicsTools/Utilities/interface/EventSelector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "PhysicsTools/PatUtils/interface/ElectronVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/PatUtils/interface/MuonVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/PatUtils/interface/JetIDSelectionFunctor.h"

class WPlusJetsEventSelector : public EventSelector {
 public:
  WPlusJetsEventSelector(
   edm::InputTag const & muonTag,
   edm::InputTag const & electronTag,
   edm::InputTag const & jetTag,
   edm::InputTag const & metTag,
   edm::InputTag const & trigTag,
   boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdTight,
   boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdTight,
   boost::shared_ptr<JetIDSelectionFunctor> & jetIdTight,
   boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdLoose,
   boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdLoose,
   boost::shared_ptr<JetIDSelectionFunctor> & jetIdLoose,
   int minJets,
   bool muPlusJets,
   bool ePlusJets,
   double muPtMin       , double muEtaMax,
   double elePtMin      , double eleEtaMax,
   double muPtMinLoose  , double muEtaMaxLoose,
   double elePtMinLoose , double eleEtaMaxLoose,
   double jetPtMin      , double jetEtaMax
			  );
  
  virtual bool operator()( edm::EventBase const & t, std::strbitset & ret);

  std::vector<pat::Jet>      const & selectedJets     () const { return selectedJets_;     } 
  std::vector<pat::Jet>      const & cleanedJets      () const { return cleanedJets_;      } 
  std::vector<pat::Electron> const & selectedElectrons() const { return selectedElectrons_;}
  std::vector<pat::Muon>     const & selectedMuons    () const { return selectedMuons_;    }
 
 protected: 

  edm::InputTag               muonTag_;
  edm::InputTag               electronTag_;
  edm::InputTag               jetTag_;
  edm::InputTag               metTag_;
  edm::InputTag               trigTag_;

  std::vector<pat::Jet>       selectedJets_;
  std::vector<pat::Muon>      selectedMuons_;
  std::vector<pat::Electron>  selectedElectrons_;
  std::vector<pat::Muon>      looseMuons_;
  std::vector<pat::Electron>  looseElectrons_;
  std::vector<pat::MET>       selectedMETs_;
  std::vector<pat::Jet>       cleanedJets_;
  std::vector<pat::Electron>  selectedElectrons2_;

  boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor>      muonIdTight_;
  boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor>  electronIdTight_;
  boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor>  electronIdTight2_;
  boost::shared_ptr<JetIDSelectionFunctor>                jetIdTight_;
  boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor>      muonIdLoose_;
  boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor>  electronIdLoose_;
  boost::shared_ptr<JetIDSelectionFunctor>                jetIdLoose_;

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


  
};


#endif
