
#include "PhysicsTools/SelectorUtils/interface/WPlusJetsEventSelector.h"

#include <iostream>

using namespace std;

WPlusJetsEventSelector::WPlusJetsEventSelector( edm::ParameterSet const & params ) :
  EventSelector(),
  muonTag_         (params.getParameter<edm::InputTag>("muonSrc") ),
  electronTag_     (params.getParameter<edm::InputTag>("electronSrc") ),
  jetTag_          (params.getParameter<edm::InputTag>("jetSrc") ),
  metTag_          (params.getParameter<edm::InputTag>("metSrc") ),  
  trigTag_         (params.getParameter<edm::InputTag>("trigSrc") ),
  muTrig_          (params.getParameter<std::string>("muTrig")),
  eleTrig_         (params.getParameter<std::string>("eleTrig")),
  pvSelector_      (params.getParameter<edm::ParameterSet>("pvSelector") ),
  muonIdTight_     (params.getParameter<edm::ParameterSet>("muonIdTight") ),
  electronIdTight_ (params.getParameter<edm::ParameterSet>("electronIdTight") ),
  muonIdLoose_     (params.getParameter<edm::ParameterSet>("muonIdLoose") ),
  electronIdLoose_ (params.getParameter<edm::ParameterSet>("electronIdLoose") ),
  jetIdLoose_      (params.getParameter<edm::ParameterSet>("jetIdLoose") ),
  pfjetIdLoose_    (params.getParameter<edm::ParameterSet>("pfjetIdLoose") ),
  minJets_         (params.getParameter<int> ("minJets") ),
  muPlusJets_      (params.getParameter<bool>("muPlusJets") ),
  ePlusJets_       (params.getParameter<bool>("ePlusJets") ),
  muPtMin_         (params.getParameter<double>("muPtMin")), 
  muEtaMax_        (params.getParameter<double>("muEtaMax")), 
  eleEtMin_        (params.getParameter<double>("eleEtMin")), 
  eleEtaMax_       (params.getParameter<double>("eleEtaMax")), 
  muPtMinLoose_    (params.getParameter<double>("muPtMinLoose")), 
  muEtaMaxLoose_   (params.getParameter<double>("muEtaMaxLoose")), 
  eleEtMinLoose_   (params.getParameter<double>("eleEtMinLoose")), 
  eleEtaMaxLoose_  (params.getParameter<double>("eleEtaMaxLoose")), 
  jetPtMin_        (params.getParameter<double>("jetPtMin")), 
  jetEtaMax_       (params.getParameter<double>("jetEtaMax")), 
  jetScale_        (params.getParameter<double>("jetScale")),
  metMin_          (params.getParameter<double>("metMin"))
{
  // make the bitset
  push_back( "Inclusive"      );
  push_back( "Trigger"        );
  push_back( "PV"             );
  push_back( ">= 1 Lepton"    );
  push_back( "== 1 Tight Lepton"    );
  push_back( "== 1 Tight Lepton, Mu Veto");
  push_back( "== 1 Lepton"    );
  push_back( "MET Cut"        );
  push_back( "Z Veto"         );
  push_back( "Conversion Veto");
  push_back( "Cosmic Veto"    );
  push_back( ">=1 Jets"       );
  push_back( ">=2 Jets"       );
  push_back( ">=3 Jets"       );
  push_back( ">=4 Jets"       );
  push_back( ">=5 Jets"       );


  // turn (almost) everything on by default
  set( "Inclusive"      );
  set( "Trigger"        );
  set( "PV"             );
  set( ">= 1 Lepton"    );
  set( "== 1 Tight Lepton"    );
  set( "== 1 Tight Lepton, Mu Veto");
  set( "== 1 Lepton"    );
  set( "MET Cut"        );
  set( "Z Veto"         );
  set( "Conversion Veto");
  set( "Cosmic Veto"    );
  set( ">=1 Jets", minJets_ >= 1);
  set( ">=2 Jets", minJets_ >= 2);
  set( ">=3 Jets", minJets_ >= 3);
  set( ">=4 Jets", minJets_ >= 4);
  set( ">=5 Jets", minJets_ >= 5); 

  dR_ = 0.3;
  muJetDR_ = 0.3;

  if ( params.exists("cutsToIgnore") )
    setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );
	

  retInternal_ = getBitTemplate();
}

bool WPlusJetsEventSelector::operator() ( edm::EventBase const & event, pat::strbitset & ret)
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


  bool passTrig = false;
  if (!ignoreCut("Trigger") ) {


    edm::Handle<pat::TriggerEvent> triggerEvent;
    event.getByLabel(trigTag_, triggerEvent);

    pat::TriggerEvent const * trig = &*triggerEvent;

    if ( trig->wasRun() && trig->wasAccept() ) {

      pat::TriggerPath const * muPath = trig->path(muTrig_);

      pat::TriggerPath const * elePath = trig->path(eleTrig_);

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


    bool passPV = false;

    passPV = pvSelector_( event );
    if ( ignoreCut("PV") || passPV ) {
      passCut(ret, "PV");
  
      edm::Handle< vector< pat::Electron > > electronHandle;
      event.getByLabel (electronTag_, electronHandle);
  
      edm::Handle< vector< pat::Muon > > muonHandle;
      event.getByLabel (muonTag_, muonHandle);

      edm::Handle< vector< pat::Jet > > jetHandle;
      event.getByLabel (jetTag_, jetHandle);

      edm::Handle< vector< pat::MET > > metHandle;
      event.getByLabel (metTag_, metHandle);

      int nElectrons = 0;
      for ( std::vector<pat::Electron>::const_iterator electronBegin = electronHandle->begin(),
	      electronEnd = electronHandle->end(), ielectron = electronBegin;
	    ielectron != electronEnd; ++ielectron ) {
	++nElectrons;
	// Tight cuts
	if ( ielectron->et() > eleEtMin_ && fabs(ielectron->eta()) < eleEtaMax_ && 
	     electronIdTight_(*ielectron) &&
	     ielectron->electronID( "eidRobustTight" ) > 0  ) {
	  selectedElectrons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Electron>( electronHandle, ielectron - electronBegin ) ) );
	} else {
	  // Loose cuts
	  if ( ielectron->et() > eleEtMinLoose_ && fabs(ielectron->eta()) < eleEtaMaxLoose_ && 
	       electronIdLoose_(*ielectron) ) {
	    looseElectrons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Electron>( electronHandle, ielectron - electronBegin ) ) );
	  }
	}
      }


      met_ = reco::ShallowClonePtrCandidate( edm::Ptr<pat::MET>( metHandle, 0),
					     metHandle->at(0).charge(),
					     metHandle->at(0).p4() );


      pat::strbitset ret1 = jetIdLoose_.getBitTemplate();
      pat::strbitset ret2 = pfjetIdLoose_.getBitTemplate();
      for ( std::vector<pat::Jet>::const_iterator jetBegin = jetHandle->begin(),
	      jetEnd = jetHandle->end(), ijet = jetBegin;
	    ijet != jetEnd; ++ijet ) {

	reco::ShallowClonePtrCandidate scaledJet ( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Jet>( jetHandle, ijet - jetBegin ),
										   ijet->charge(),
										   ijet->p4() * jetScale_ ) );
    
	bool passJetID = false;
	if ( ijet->isCaloJet() ) passJetID = jetIdLoose_(*ijet, ret1);
	else passJetID = pfjetIdLoose_(*ijet, ret2);
	if ( scaledJet.pt() > jetPtMin_ && fabs(scaledJet.eta()) < jetEtaMax_ && passJetID ) {
	  selectedJets_.push_back( scaledJet );
	  if ( muPlusJets_ ) {
	    cleanedJets_.push_back( scaledJet );
	  } else {
	    //Remove some jets
	    bool indeltaR = false;
	    for( std::vector<reco::ShallowClonePtrCandidate>::const_iterator electronBegin = selectedElectrons_.begin(),
		   electronEnd = selectedElectrons_.end(), ielectron = electronBegin;
		 ielectron != electronEnd; ++ielectron ) {
	      if( reco::deltaR( ielectron->eta(), ielectron->phi(), scaledJet.eta(), scaledJet.phi() ) < dR_ )
		{  indeltaR = true; }
	    }
	    if( !indeltaR ) {
	      cleanedJets_.push_back( scaledJet );
	    }
	  }
	}
      }

      for ( std::vector<pat::Muon>::const_iterator muonBegin = muonHandle->begin(),
	      muonEnd = muonHandle->end(), imuon = muonBegin;
	    imuon != muonEnd; ++imuon ) {
	if ( !imuon->isGlobalMuon() ) continue;

	//Now, check that the muon isn't within muJetDR_ of any jet
	bool inDeltaR = false;
	for (std::vector<reco::ShallowClonePtrCandidate>::const_iterator iJet = selectedJets_.begin();
	     iJet != selectedJets_.end(); ++iJet) {
	  if ( reco::deltaR(imuon->eta(), imuon->phi(), iJet->eta(), iJet->phi()) < muJetDR_ ) inDeltaR = true;
	}
	
	// Tight cuts
	bool passTight = muonIdTight_(*imuon,event) && imuon->isTrackerMuon() && !inDeltaR;
	if (  imuon->pt() > muPtMin_ && fabs(imuon->eta()) < muEtaMax_ && 
	     passTight ) {

	  selectedMuons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Muon>( muonHandle, imuon - muonBegin ) ) );
	} else {
	  // Loose cuts
	  if ( imuon->pt() > muPtMinLoose_ && fabs(imuon->eta()) < muEtaMaxLoose_ && 
	       muonIdLoose_(*imuon,event) ) {
	    looseMuons_.push_back( reco::ShallowClonePtrCandidate( edm::Ptr<pat::Muon>( muonHandle, imuon - muonBegin ) ) );
	  }
	}
      }





      int nleptons = 0;
      if ( muPlusJets_ )
	nleptons += selectedMuons_.size();
      
      if ( ePlusJets_ ) 
	nleptons += selectedElectrons_.size();

      if ( ignoreCut(">= 1 Lepton") || 
	   ( nleptons > 0 ) ){
	passCut( ret, ">= 1 Lepton");

	if ( ignoreCut("== 1 Tight Lepton") || 
	     ( nleptons == 1 ) ){
	  passCut( ret, "== 1 Tight Lepton");

	  bool oneMuon = 
	    ( selectedMuons_.size() == 1 && 
	      looseMuons_.size() + selectedElectrons_.size() + looseElectrons_.size() == 0 
	      );
	  bool oneElectron = 
	    ( selectedElectrons_.size() == 1 &&
	      selectedMuons_.size() == 0 
	      );

	  bool oneMuonMuVeto = 
	    ( selectedMuons_.size() == 1 && 
	      looseMuons_.size() == 0 
	      );


	  if ( ignoreCut("== 1 Tight Lepton, Mu Veto") || 
	       ( (muPlusJets_ && oneMuonMuVeto)  )
	       ) {
	    passCut(ret, "== 1 Tight Lepton, Mu Veto");

	    if ( ignoreCut("== 1 Lepton") || 
		 ( (muPlusJets_ && oneMuon) ^ (ePlusJets_ && oneElectron )  )
		 ) {
	      passCut(ret, "== 1 Lepton");	  

	      bool metCut = met_.pt() > metMin_;
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

		      if ( ignoreCut(">=1 Jets") ||
			   static_cast<int>(cleanedJets_.size()) >=  1 ){
			passCut(ret,">=1 Jets");  
		      } // end if >=1 tight jets

		      if ( ignoreCut(">=2 Jets") ||
			   static_cast<int>(cleanedJets_.size()) >=  2 ){
			passCut(ret,">=2 Jets");  
		      } // end if >=2 tight jets

		      if ( ignoreCut(">=3 Jets") ||
			   static_cast<int>(cleanedJets_.size()) >=  3 ){
			passCut(ret,">=3 Jets");  
		      } // end if >=3 tight jets

		      if ( ignoreCut(">=4 Jets") ||
			   static_cast<int>(cleanedJets_.size()) >=  4 ){
			passCut(ret,">=4 Jets");  
		      } // end if >=4 tight jets

		      if ( ignoreCut(">=5 Jets") ||
			   static_cast<int>(cleanedJets_.size()) >=  5 ){
			passCut(ret,">=5 Jets");  
		      } // end if >=5 tight jets


		  
		    } // end if cosmic veto
		
		  } // end if conversion veto

		} // end if z veto

	      } // end if met cut
	
	    } // end if == 1 lepton

	  } // end if == 1 tight lepton with a muon veto separately

	} // end if == 1 tight lepton

      } // end if >= 1 lepton

    } // end if PV
    
  } // end if trigger


  setIgnored(ret);

  return (bool)ret;
}
