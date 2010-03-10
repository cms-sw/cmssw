
#include "PhysicsTools/PatExamples/interface/WPlusJetsEventSelector.h"

#include <iostream>

using namespace std;

WPlusJetsEventSelector::WPlusJetsEventSelector( edm::ParameterSet const & params ) :
  EventSelector(),
  muonTag_         (params.getParameter<edm::InputTag>("muonSrc") ),
  electronTag_     (params.getParameter<edm::InputTag>("electronSrc") ),
  jetTag_          (params.getParameter<edm::InputTag>("jetSrc") ),
  metTag_          (params.getParameter<edm::InputTag>("metSrc") ),  
  trigTag_         (params.getParameter<edm::InputTag>("trigSrc") ),  
  muonIdTight_     (params.getParameter<edm::ParameterSet>("muonIdTight") ),
  electronIdTight_ (params.getParameter<edm::ParameterSet>("electronIdTight") ),
  muonIdLoose_     (params.getParameter<edm::ParameterSet>("muonIdLoose") ),
  electronIdLoose_ (params.getParameter<edm::ParameterSet>("electronIdLoose") ),
  jetIdLoose_      (params.getParameter<edm::ParameterSet>("jetIdLoose") ),
  minJets_         (params.getParameter<int> ("minJets") ),
  muPlusJets_      (params.getParameter<bool>("muPlusJets") ),
  ePlusJets_       (params.getParameter<bool>("ePlusJets") ),
  muPtMin_         (params.getParameter<double>("muPtMin")), 
  muEtaMax_        (params.getParameter<double>("muEtaMax")), 
  elePtMin_        (params.getParameter<double>("elePtMin")), 
  eleEtaMax_       (params.getParameter<double>("eleEtaMax")), 
  muPtMinLoose_    (params.getParameter<double>("muPtMinLoose")), 
  muEtaMaxLoose_   (params.getParameter<double>("muEtaMaxLoose")), 
  elePtMinLoose_   (params.getParameter<double>("elePtMinLoose")), 
  eleEtaMaxLoose_  (params.getParameter<double>("eleEtaMaxLoose")), 
  jetPtMin_        (params.getParameter<double>("jetPtMin")), 
  jetEtaMax_       (params.getParameter<double>("jetEtaMax")), 
  jetScale_        (params.getParameter<double>("jetScale"))
{
  // make the bitset
  push_back( "Inclusive"      );
  push_back( "Trigger"        );
  push_back( ">= 1 Lepton"    );
  push_back( "== 1 Lepton"    );
  push_back( "Tight Jet Cuts", minJets_ );
  push_back( "MET Cut"        );
  push_back( "Z Veto"         );
  push_back( "Conversion Veto");
  push_back( "Cosmic Veto"    );

  // turn everything on by default
  set( "Inclusive"      );
  set( "Trigger"        );
  set( ">= 1 Lepton"    );
  set( "== 1 Lepton"    );
  set( "Tight Jet Cuts" );
  set( "MET Cut"        );
  set( "Z Veto"         );
  set( "Conversion Veto");
  set( "Cosmic Veto"    );

  dR_ = 0.3;

  retInternal_ = getBitTemplate();
}

bool WPlusJetsEventSelector::operator() ( edm::EventBase const & event, std::strbitset & ret)
{
  ret.set(false);

  selectedJets_.clear();
  cleanedJets_.clear();
  selectedMuons_.clear();
  selectedElectrons_.clear();
  looseMuons_.clear();
  looseElectrons_.clear();
  selectedMETs_.clear();


  passCut( ret, "Inclusive");
  
  edm::Handle< vector< pat::Electron > > electronHandle;
  event.getByLabel (electronTag_, electronHandle);
  
  edm::Handle< vector< pat::Muon > > muonHandle;
  event.getByLabel (muonTag_, muonHandle);

  edm::Handle< vector< pat::Jet > > jetHandle;
  event.getByLabel (jetTag_, jetHandle);

  edm::Handle< vector< pat::MET > > metHandle;
  event.getByLabel (metTag_, metHandle);

  edm::Handle<pat::TriggerEvent> triggerEvent;
  event.getByLabel(trigTag_, triggerEvent);


  int nGlobalMuons = 0;
  for ( std::vector<pat::Muon>::const_iterator muonBegin = muonHandle->begin(),
	  muonEnd = muonHandle->end(), imuon = muonBegin;
	imuon != muonEnd; ++imuon ) {
    if ( imuon->isGlobalMuon() ) {
      ++nGlobalMuons;
      // Tight cuts
      bool passTight = muonIdTight_(*imuon);
      if ( imuon->pt() > muPtMin_ && fabs(imuon->eta()) < muEtaMax_ && 
	   passTight ) {
	selectedMuons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Muon>( muonHandle, imuon - muonBegin ) ) );
      } else {
	// Loose cuts
	if ( imuon->pt() > muPtMinLoose_ && fabs(imuon->eta()) < muEtaMaxLoose_ && 
	     muonIdLoose_(*imuon) ) {
	  looseMuons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Muon>( muonHandle, imuon - muonBegin ) ) );
	}
      }
    }
  }

  int nElectrons = 0;
  for ( std::vector<pat::Electron>::const_iterator electronBegin = electronHandle->begin(),
	  electronEnd = electronHandle->end(), ielectron = electronBegin;
	ielectron != electronEnd; ++ielectron ) {
    ++nElectrons;
    // Tight cuts
    if ( ielectron->pt() > elePtMin_ && fabs(ielectron->eta()) < eleEtaMax_ && 
	 electronIdTight_(*ielectron) &&
	 ielectron->electronID( "eidRobustTight" ) > 0  ) {
      selectedElectrons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Electron>( electronHandle, ielectron - electronBegin ) ) );
    } else {
      // Loose cuts
      if ( ielectron->pt() > elePtMinLoose_ && fabs(ielectron->eta()) < eleEtaMaxLoose_ && 
	   electronIdLoose_(*ielectron) ) {
	looseElectrons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Electron>( electronHandle, ielectron - electronBegin ) ) );
      }
    }
  }

  for ( std::vector<pat::Jet>::const_iterator jetBegin = jetHandle->begin(),
	  jetEnd = jetHandle->end(), ijet = jetBegin;
	ijet != jetEnd; ++ijet ) {

    reco::ShallowClonePtrCandidate scaledJet ( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Jet>( jetHandle, ijet - jetBegin ),
									       ijet->charge(),
									       ijet->p4() * jetScale_ ) );
    
    if ( scaledJet.pt() > jetPtMin_ && fabs(scaledJet.eta()) < jetEtaMax_ && jetIdLoose_(*ijet) ) {
      selectedJets_.push_back( scaledJet );
      if ( muPlusJets_ ) {
	cleanedJets_.push_back( scaledJet );
      } else {
	//Remove some jets
	bool indeltaR = false;
	for( std::vector<reco::ShallowClonePtrCandidate>::const_iterator electronBegin = selectedElectrons_.begin(),
	       electronEnd = selectedElectrons_.end(), ielectron = electronBegin;
	     ielectron != electronEnd; ++ielectron )
	  if( reco::deltaR( ielectron->eta(), ielectron->phi(), scaledJet.eta(), scaledJet.phi() ) < dR_ )
	    {  indeltaR = true;  continue; }
	if( !indeltaR ) {
	  cleanedJets_.push_back( scaledJet );
	}
      }
    }
  }


  bool passTrig = false;
  if (!ignoreCut("Trigger") ) {

    pat::TriggerEvent const * trig = &*triggerEvent;

    if ( trig->wasRun() && trig->wasAccept() ) {

      pat::TriggerPath const * muPath = trig->path("HLT_Mu9");

      pat::TriggerPath const * elePath = trig->path("HLT_Ele15_LW_L1R");

      if ( muPlusJets_ && muPath != 0 && muPath->wasAccept() ) {
	passTrig = true;    
      }
      
      if ( ePlusJets_ && elePath != 0 && elePath->wasAccept() ) {
	passTrig = true;
      }
    }
  }



  
  if ( ignoreCut("Trigger") || 
       passTrig ) {
    passCut(ret, "Trigger");

    int nleptons = 0;
    if ( muPlusJets_ )
      nleptons += selectedMuons_.size();
      
    if ( ePlusJets_ ) 
      nleptons += selectedElectrons_.size();

    if ( ignoreCut(">= 1 Lepton") || 
	 ( nleptons > 0 ) ){
      passCut( ret, ">= 1 Lepton");

      bool oneMuon = 
	( selectedMuons_.size() == 1 && 
	  looseMuons_.size() + selectedElectrons_.size() + looseElectrons_.size() == 0 
	  );
      bool oneElectron = 
	( selectedElectrons_.size() == 1 &&
	  selectedMuons_.size() == 0 
	  );

      if ( ignoreCut("== 1 Lepton") || 
	   ( (muPlusJets_ && oneMuon) ^ (ePlusJets_ && oneElectron )  )
	   ) {
	passCut(ret, "== 1 Lepton");

	if ( ignoreCut("Tight Jet Cuts") ||
	     static_cast<int>(cleanedJets_.size()) >=  this->cut("Tight Jet Cuts", int()) ){
	  passCut(ret,"Tight Jet Cuts");
	  

	  bool metCut = true;
	  if ( ignoreCut("MET Cut") ||
	       metCut ) {
	    passCut( ret, "MET Cut" );
	  

	    bool zVeto = true;
	    if ( selectedMuons_.size() == 2 ) {
	    }
	    if ( selectedElectrons_.size() == 2 ) {
	    }
	    if ( ignoreCut("Z Veto") ||
		 zVeto ){
	      passCut(ret, "Z Veto");
	    
  
	      bool conversionVeto = true;
	      if ( ignoreCut("Conversion Veto") ||
		   conversionVeto ) {
		passCut(ret,"Conversion Veto");
		


		bool cosmicVeto = true;
		if ( ignoreCut("Cosmic Veto") ||
		     cosmicVeto ) {
		  passCut(ret,"Cosmic Veto");

		  
		} // end if cosmic veto
		
	      } // end if conversion veto

	    } // end if z veto

	  } // end if met cut
      
	} // end if 1 tight jet 
	
      } // end if == 1 lepton

    } // end if >= 1 lepton
    
  } // end if trigger


  setIgnored(ret);

  return (bool)ret;
}
