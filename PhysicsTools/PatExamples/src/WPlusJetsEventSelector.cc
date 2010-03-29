
#include "PhysicsTools/PatExamples/interface/WPlusJetsEventSelector.h"

#include <iostream>

using namespace std;

WPlusJetsEventSelector::WPlusJetsEventSelector( 
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
    ) :
  EventSelector(),
  muonTag_(muonTag),
  electronTag_(electronTag),
  jetTag_(jetTag),
  metTag_(metTag),
  trigTag_(trigTag),
  muonIdTight_(muonIdTight),
  electronIdTight_(electronIdTight),
  jetIdTight_(jetIdTight),
  muonIdLoose_(muonIdLoose),
  electronIdLoose_(electronIdLoose),
  jetIdLoose_(jetIdLoose),
  minJets_(minJets),
  muPlusJets_(muPlusJets),
  ePlusJets_(ePlusJets),
  muPtMin_(muPtMin), muEtaMax_(muEtaMax),
  elePtMin_(elePtMin), eleEtaMax_(eleEtaMax),
  muPtMinLoose_(muPtMinLoose), muEtaMaxLoose_(muEtaMaxLoose),
  elePtMinLoose_(elePtMinLoose), eleEtaMaxLoose_(eleEtaMaxLoose),
  jetPtMin_(jetPtMin), jetEtaMax_(jetEtaMax)
{
  // make the bitset
  push_back( "Inclusive"      );
  push_back( "Trigger"        );
  push_back( ">= 1 Lepton"    );
  push_back( "== 1 Lepton"    );
  push_back( "Tight Jet Cuts", minJets );
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
  event.getByLabel(edm::InputTag("patTriggerEvent"), triggerEvent);


  int nGlobalMuons = 0;
  for ( std::vector<pat::Muon>::const_iterator muonBegin = muonHandle->begin(),
	  muonEnd = muonHandle->end(), imuon = muonBegin;
	imuon != muonEnd; ++imuon ) {
    if ( imuon->isGlobalMuon() ) {
      ++nGlobalMuons;
      // Tight cuts
      std::strbitset iret = muonIdTight_->getBitTemplate();
      if ( imuon->pt() > muPtMin_ && fabs(imuon->eta()) < muEtaMax_ && 
	   (*muonIdTight_)(*imuon, iret) ) {
	selectedMuons_.push_back( *imuon );
      } else {
	// Loose cuts
	std::strbitset iret = muonIdLoose_->getBitTemplate();
	if ( imuon->pt() > muPtMinLoose_ && fabs(imuon->eta()) < muEtaMaxLoose_ && 
	     (*muonIdLoose_)(*imuon, iret) ) {
	  looseMuons_.push_back( *imuon );
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
    std::strbitset iret = electronIdTight_->getBitTemplate();
    if ( ielectron->pt() > elePtMin_ && fabs(ielectron->eta()) < eleEtaMax_ && 
	 (*electronIdTight_)(*ielectron, iret) &&
	 ielectron->electronID( "eidRobustTight" ) > 0  ) {
      selectedElectrons_.push_back( *ielectron );
    } else {
      // Loose cuts
      std::strbitset iret = electronIdLoose_->getBitTemplate();
      if ( ielectron->pt() > elePtMinLoose_ && fabs(ielectron->eta()) < eleEtaMaxLoose_ && 
	   (*electronIdLoose_)(*ielectron, iret) ) {
	looseElectrons_.push_back( *ielectron );
      }
    }
  }

  for ( std::vector<pat::Jet>::const_iterator jetBegin = jetHandle->begin(),
	  jetEnd = jetHandle->end(), ijet = jetBegin;
	ijet != jetEnd; ++ijet ) {
    std::strbitset iret = jetIdTight_->getBitTemplate();
//     if ( ijet->pt() > jetPtMin_ && fabs(ijet->eta()) < jetEtaMax_ && (*jetIdTight_)(*ijet, iret) ) {
    if ( ijet->pt() > jetPtMin_ && fabs(ijet->eta()) < jetEtaMax_ ) {
      selectedJets_.push_back( *ijet );
      if ( muPlusJets_ ) {
	cleanedJets_.push_back( *ijet );
      } else {
	//Remove some jets
	bool indeltaR = false;
	for( std::vector<pat::Electron>::const_iterator electronBegin = selectedElectrons_.begin(),
	       electronEnd = selectedElectrons_.end(), ielectron = electronBegin;
	     ielectron != electronEnd; ++ielectron )
	  if( reco::deltaR( ielectron->eta(), ielectron->phi(), ijet->eta(), ijet->phi() ) < dR_ )
	    {  indeltaR = true;  continue; }
	if( !indeltaR ) {
	  cleanedJets_.push_back( *ijet );
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
