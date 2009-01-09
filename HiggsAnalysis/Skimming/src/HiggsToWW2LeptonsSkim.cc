/** \class HiggsToWW2LeptonsSkim
 *
 *  
 *  This class is an EDFilter for HWW events
 *
 *  $Date: 2007/12/20 00:40:30 $
 *  $Revision: 1.10 $
 *
 *  \author Ezio Torassa  -  INFN Padova
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToWW2LeptonsSkim.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

// Muons:
#include <DataFormats/TrackReco/interface/Track.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

HiggsToWW2LeptonsSkim::HiggsToWW2LeptonsSkim(const edm::ParameterSet& iConfig) :
  nEvents_(0), nAccepted_(0)
{

  // Reconstructed objects
  recTrackLabel     = iConfig.getParameter<edm::InputTag>("RecoTrackLabel");
  theGLBMuonLabel   = iConfig.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
  theGsfELabel      = iConfig.getParameter<edm::InputTag>("ElectronCollectionLabel");

  singleTrackPtMin_ = iConfig.getParameter<double>("SingleTrackPtMin");
  diTrackPtMin_     = iConfig.getParameter<double>("DiTrackPtMin");
  etaMin_           = iConfig.getParameter<double>("etaMin");
  etaMax_           = iConfig.getParameter<double>("etaMax");
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
  int nTrackOver2ndCut = 0;


  // Handle<CandidateCollection> tracks;

  using reco::TrackCollection;

  // Get the muon track collection from the event
  edm::Handle<reco::TrackCollection> muTracks;
  event.getByLabel(theGLBMuonLabel.label(), muTracks);
  
  if ( muTracks.isValid() ) {

    reco::TrackCollection::const_iterator muons;

    // Loop over muon collections and count how many muons there are,
    // and how many are above threshold
    for ( muons = muTracks->begin(); muons != muTracks->end(); ++muons ) {
      if ( muons->eta() > etaMin_ && muons->eta() < etaMax_ ) {
        if ( muons->pt() > singleTrackPtMin_ ) accepted1 = true;
        if ( muons->pt() > diTrackPtMin_ ) nTrackOver2ndCut++; 
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
        if ( electrons->pt() > singleTrackPtMin_ ) accepted1 = true;
        if ( electrons->pt() > diTrackPtMin_ ) nTrackOver2ndCut++;
      }
    }
  }


  if ( accepted1 && nTrackOver2ndCut >= 2 ) accepted = true;

  if ( accepted ) nAccepted_++;

  return accepted;

}
