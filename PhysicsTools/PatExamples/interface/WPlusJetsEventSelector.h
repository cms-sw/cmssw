#ifndef PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h
#define PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h

#include "PhysicsTools/Utilities/interface/EventSelector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "PhysicsTools/PatUtils/interface/ElectronVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/PatUtils/interface/MuonVPlusJetsIDSelectionFunctor.h"
#include "PhysicsTools/PatUtils/interface/JetIDSelectionFunctor.h"

class WPlusJetsEventSelector : public EventSelector {
 public:
  WPlusJetsEventSelector( 
   edm::InputTag const & muSrc, 
   edm::InputTag const & eleSrc,
   edm::InputTag const & jetSrc,
   edm::InputTag const & metSrc,
   boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdTight,
   boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdTight,
   boost::shared_ptr<JetIDSelectionFunctor> & jetIdTight,
   boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor> & muonIdLoose,
   boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor> & electronIdLoose,
   boost::shared_ptr<JetIDSelectionFunctor> & jetIdLoose,
   double muPtMin  = 20.0,
   double elePtMin = 20.0,
   double jetPtMin = 30.0
			  );
  
  virtual bool operator()( edm::EventBase const & t, std::strbitset & ret);

  std::vector<pat::Jet>      const & selectedJets     () const { return selectedJets_;     } 
  std::vector<pat::Electron> const & selectedElectrons() const { return selectedElectrons_;}
  std::vector<pat::Muon>     const & selectedMuons    () const { return selectedMuons_;    }
 
 protected: 
  edm::InputTag muSrc_;
  edm::InputTag eleSrc_;
  edm::InputTag jetSrc_;
  edm::InputTag metSrc_;

  std::vector<pat::Jet>       selectedJets_;
  std::vector<pat::Muon>      selectedMuons_;
  std::vector<pat::Electron>  selectedElectrons_;
  std::vector<pat::MET>       selectedMETs_;

  boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor>      muonIdTight_;
  boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor>  electronIdTight_;
  boost::shared_ptr<JetIDSelectionFunctor>                jetIdTight_;
  boost::shared_ptr<MuonVPlusJetsIDSelectionFunctor>      muonIdLoose_;
  boost::shared_ptr<ElectronVPlusJetsIDSelectionFunctor>  electronIdLoose_;
  boost::shared_ptr<JetIDSelectionFunctor>                jetIdLoose_;

  double muPtMin_  ;
  double elePtMin_ ;
  double jetPtMin_ ;
  
};


#endif
