/** \class HiggsToWW2LeptonsSkim
 *
 *  
 *  This class is an EDFilter for HWW events
 *
 *  $Date: 2009/08/17 20:32:33 $
 *  $Revision: 1.17 $
 *
 *  \author Ezio Torassa  -  INFN Padova
 *  \revised Javier Fernandez - Univ. Oviedo	
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToWW2LeptonsSkim.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

//Tracks
#include <DataFormats/TrackReco/interface/Track.h>

// Muons:
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muon;

HiggsToWW2LeptonsSkim::HiggsToWW2LeptonsSkim(const edm::ParameterSet& iConfig) :
  nEvents_(0), nAccepted_(0)
{

  // Reconstructed objects
  theMuonLabel   = iConfig.getParameter<edm::InputTag>("MuonCollectionLabel");
  theGsfELabel      = iConfig.getParameter<edm::InputTag>("ElectronCollectionLabel");

  nLeptons_	     = iConfig.getParameter<int>("nLeptons");
  if(nLeptons_>2) {nLeptons_=2;
    edm::LogVerbatim("HiggsToWW2LeptonsSkim")
            << "\nnLeptons_ must be 1 or 2!!!"
            << std::endl;
	}
  singleLeptonPtMin_ = iConfig.getParameter<double>("SingleLeptonPtMin");
  diLeptonPtMin_     = iConfig.getParameter<double>("DiLeptonPtMin");
  etaMin_           = iConfig.getParameter<double>("etaMin");
  etaMax_           = iConfig.getParameter<double>("etaMax");
  muonType_	    = iConfig.getParameter<string>("muonType");
 
  beTight_	    = iConfig.getParameter<bool>("beTight");
  dilepM_	    = iConfig.getParameter<double>("dilepM");
  eleHadronicOverEm_= iConfig.getParameter<double>("eleHadronicOverEm");
}


HiggsToWW2LeptonsSkim::~HiggsToWW2LeptonsSkim()
{
}

void HiggsToWW2LeptonsSkim::endJob() 
{
  edm::LogVerbatim("HiggsToWW2LeptonsSkim") 
	    << "Events read " << nEvents_ 
            << " Events accepted " << nAccepted_ 
            << "\nEfficiency " << ((double)nAccepted_)/((double)nEvents_) 
	    << std::endl;
}


bool HiggsToWW2LeptonsSkim::filter(edm::Event& event, const edm::EventSetup& iSetup)
{

  nEvents_++;
  bool accepted = false;
  bool accepted1 = false;
  int nLeptonOver2ndCut = 0;
  std::vector<Particle::LorentzVector> leptons;
  double MuMass = 0.106;
  double EMass=0.000511;


  // Handle<CandidateCollection> tracks;

  using reco::TrackCollection;
  using reco::MuonCollection;

  // Get the muon track collection from the event
  edm::Handle<reco::MuonCollection> muTracks;
  event.getByLabel(theMuonLabel.label(), muTracks);
  
  if ( muTracks.isValid() ) {

    reco::MuonCollection::const_iterator muons;

  
    // Loop over muon collections and count how many muons there are,
    // and how many are above threshold
    for ( muons = muTracks->begin(); muons != muTracks->end(); ++muons ) {
    if(muon::isGoodMuon(*muons,muon::selectionTypeFromString(muonType_))){  
    if ( muons->eta() > etaMin_ && muons->eta() < etaMax_ ) {
        if ( muons->pt() > singleLeptonPtMin_ ) accepted1 = true;
        if ( muons->pt() > diLeptonPtMin_ ) nLeptonOver2ndCut++;
	if(beTight_){
	double E = sqrt(muons->momentum().Mag2()+MuMass*MuMass);
	leptons.push_back(Particle::LorentzVector(muons->px(),muons->py(),muons->pz(),E));
	}
      }
    }
  } 
  }

  // Now look at electrons:

  // Get the electron track collection from the event
  edm::Handle<reco::GsfElectronCollection> pTracks;

  event.getByLabel(theGsfELabel.label(),pTracks);

  if ( pTracks.isValid() ) {

    const reco::GsfElectronCollection* eTracks = pTracks.product();
   
    reco::GsfElectronCollection::const_iterator electrons;

    // Loop over electron collections and count how many muons there are,
    // and how many are above threshold
    for ( electrons = eTracks->begin(); electrons != eTracks->end(); ++electrons ) {
      if ( electrons->eta() > etaMin_ && electrons->eta() < etaMax_ ) {

	if(beTight_ && electrons->hadronicOverEm() < eleHadronicOverEm_){
        if ( electrons->pt() > singleLeptonPtMin_ ) accepted1 = true;
        if ( electrons->pt() > diLeptonPtMin_ ) nLeptonOver2ndCut++;
        double e = sqrt(electrons->momentum().Mag2()+EMass*EMass);
        leptons.push_back(Particle::LorentzVector(electrons->px(),electrons->py(),electrons->pz(),e));
	}
	else{
        if ( electrons->pt() > singleLeptonPtMin_ ) accepted1 = true;
        if ( electrons->pt() > diLeptonPtMin_ ) nLeptonOver2ndCut++;
  	}
		
     }
    }
  }


  if ( accepted1 && nLeptonOver2ndCut >= nLeptons_ ) accepted = true;

  if(accepted && beTight_){
	accepted=false;
	if(leptons.size()>0){
		for(unsigned int i=0;i<leptons.size();i++){
			for(unsigned int j=i+1;j<leptons.size();j++){
			Particle::LorentzVector lep1 = leptons.at(i);
			Particle::LorentzVector lep2 = leptons.at(j);
			Particle::LorentzVector lep = lep1 + lep2;
             		double invmass = abs(lep.mass());
			if(invmass>dilepM_){
			accepted=true;
			break;
			}
			}
		}
	}
  }

  if ( accepted ) nAccepted_++;

  return accepted;

}
