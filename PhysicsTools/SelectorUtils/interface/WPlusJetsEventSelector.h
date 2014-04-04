#ifndef PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h
#define PhysicsTools_PatExamples_interface_WPlusJetsEventSelector_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
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
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PVSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

class WPlusJetsEventSelector : public EventSelector {
 public:
  WPlusJetsEventSelector() {}
#ifndef __GCCXML__
  WPlusJetsEventSelector( edm::ParameterSet const & params, edm::ConsumesCollector&& iC ) :
    WPlusJetsEventSelector(params)
  {
    muonToken_ = iC.mayConsume< std::vector< pat::Muon > >(muonTag_);
    electronToken_ = iC.mayConsume< std::vector< pat::Electron > >(electronTag_);
    jetToken_ = iC.mayConsume< std::vector< pat::Jet > >(jetTag_);
    metToken_ = iC.mayConsume< std::vector< pat::MET > >(metTag_);
    trigToken_ = iC.mayConsume<pat::TriggerEvent>(trigTag_);
    muonIdTight_ = MuonVPlusJetsIDSelectionFunctor( params.getParameter<edm::ParameterSet>("muonIdTight"), iC );
    electronIdTight_ = ElectronVPlusJetsIDSelectionFunctor(params.getParameter<edm::ParameterSet>("electronIdTight"), iC );
    muonIdLoose_ = MuonVPlusJetsIDSelectionFunctor( params.getParameter<edm::ParameterSet>("muonIdLoose"), iC );
    electronIdLoose_ = ElectronVPlusJetsIDSelectionFunctor(params.getParameter<edm::ParameterSet>("electronIdLoose"), iC );
    jetIdLoose_      = JetIDSelectionFunctor(params.getParameter<edm::ParameterSet>("jetIdLoose"), iC );
    pfjetIdLoose_    = PFJetIDSelectionFunctor(params.getParameter<edm::ParameterSet>("pfjetIdLoose"), iC );
  }
#endif
  WPlusJetsEventSelector( edm::ParameterSet const & params );

  virtual void scaleJets(double scale) {jetScale_ = scale;}

  virtual bool operator()( edm::EventBase const & t, pat::strbitset & ret);
  using EventSelector::operator();

  std::vector<reco::ShallowClonePtrCandidate> const & selectedJets     () const { return selectedJets_;     }
  std::vector<reco::ShallowClonePtrCandidate> const & cleanedJets      () const { return cleanedJets_;      }
  std::vector<reco::ShallowClonePtrCandidate> const & selectedElectrons() const { return selectedElectrons_;}
  std::vector<reco::ShallowClonePtrCandidate> const & selectedMuons    () const { return selectedMuons_;    }
  reco::ShallowClonePtrCandidate const &              selectedMET      () const { return met_; }

  void printSelectors(std::ostream & out) const {
    out << "PV Selector: " << std::endl;
    pvSelector_.print(out);
    out << "Muon ID Tight Selector: " << std::endl;
    muonIdTight_.print(out);
    out << "Electron ID Tight Selector: " << std::endl;
    electronIdTight_.print(out);
    out << "Muon ID Loose Selector: " << std::endl;
    muonIdLoose_.print(out);
    out << "Electron ID Loose Selector: " << std::endl;
    electronIdLoose_.print(out);
    out << "Calo Jet Selector: " << std::endl;
    jetIdLoose_.print(out);
    out << "PF Jet Selector: " << std::endl;
    pfjetIdLoose_.print(out);
  }

 protected:

  edm::InputTag               muonTag_;
#ifndef __GCCXML__
  edm::EDGetTokenT< std::vector< pat::Muon > > muonToken_;
#endif
  edm::InputTag               electronTag_;
#ifndef __GCCXML__
  edm::EDGetTokenT< std::vector< pat::Electron > > electronToken_;
#endif
  edm::InputTag               jetTag_;
#ifndef __GCCXML__
  edm::EDGetTokenT< std::vector< pat::Jet > > jetToken_;
#endif
  edm::InputTag               metTag_;
#ifndef __GCCXML__
  edm::EDGetTokenT< std::vector< pat::MET > > metToken_;
#endif
  edm::InputTag               trigTag_;
#ifndef __GCCXML__
  edm::EDGetTokenT<pat::TriggerEvent> trigToken_;
#endif

  std::string                 muTrig_;
  std::string                 eleTrig_;

  std::vector<reco::ShallowClonePtrCandidate> selectedJets_;
  std::vector<reco::ShallowClonePtrCandidate> selectedMuons_;
  std::vector<reco::ShallowClonePtrCandidate> selectedElectrons_;
  std::vector<reco::ShallowClonePtrCandidate> looseMuons_;
  std::vector<reco::ShallowClonePtrCandidate> looseElectrons_;
  std::vector<reco::ShallowClonePtrCandidate> selectedMETs_;
  std::vector<reco::ShallowClonePtrCandidate> cleanedJets_;
  std::vector<reco::ShallowClonePtrCandidate> selectedElectrons2_;
  reco::ShallowClonePtrCandidate              met_;

  PVSelector                           pvSelector_;
  MuonVPlusJetsIDSelectionFunctor      muonIdTight_;
  ElectronVPlusJetsIDSelectionFunctor  electronIdTight_;
  MuonVPlusJetsIDSelectionFunctor      muonIdLoose_;
  ElectronVPlusJetsIDSelectionFunctor  electronIdLoose_;
  JetIDSelectionFunctor                jetIdLoose_;
  PFJetIDSelectionFunctor              pfjetIdLoose_;

  int minJets_;

  double muJetDR_;
  double eleJetDR_;

  bool muPlusJets_;
  bool ePlusJets_;

  double muPtMin_  ;
  double muEtaMax_ ;
  double eleEtMin_ ;
  double eleEtaMax_;

  double muPtMinLoose_  ;
  double muEtaMaxLoose_ ;
  double eleEtMinLoose_ ;
  double eleEtaMaxLoose_;

  double jetPtMin_ ;
  double jetEtaMax_;

  double jetScale_;

  double metMin_;

  index_type   inclusiveIndex_;
  index_type   triggerIndex_;
  index_type   pvIndex_;
  index_type   lep1Index_;
  index_type   lep2Index_;
  index_type   lep3Index_;
  index_type   lep4Index_;
  index_type   metIndex_;
  index_type   zvetoIndex_;
  index_type   conversionIndex_;
  index_type   cosmicIndex_;
  index_type   jet1Index_;
  index_type   jet2Index_;
  index_type   jet3Index_;
  index_type   jet4Index_;
  index_type   jet5Index_;

};


#endif
