/** \class HiggsToWW2LeptonsSkim
 *
 *  
 *  This class is an EDFilter for HWW events
 *
 *  $Date: 2007/08/05 22:09:26 $
 *  $Revision: 1.6 $
 *
 *  \author Ezio Torassa  -  INFN Padova
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToWW2LeptonsSkim.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

// Muons:
#include <DataFormats/TrackReco/interface/Track.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

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
  thePixelGsfELabel = iConfig.getParameter<edm::InputTag>("ElectronCollectionLabel");

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

  try {
  // Get the muon track collection from the event
    edm::Handle<reco::TrackCollection> muTracks;
    event.getByLabel(theGLBMuonLabel.label(), muTracks);
  
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
  catch (const edm::Exception& e) {
    //wrong reason for exception
    if ( e.categoryCode() != edm::errors::ProductNotFound ) throw;
  }

  // Now look at electrons:

  try {
    // Get the electron track collection from the event
    edm::Handle<reco::PixelMatchGsfElectronCollection> pTracks;

    event.getByLabel(thePixelGsfELabel.label(),pTracks);
    const reco::PixelMatchGsfElectronCollection* eTracks = pTracks.product();
   
    reco::PixelMatchGsfElectronCollection::const_iterator electrons;

    // Loop over electron collections and count how many muons there are,
    // and how many are above threshold
    for ( electrons = eTracks->begin(); electrons != eTracks->end(); ++electrons ) {
      if ( electrons->eta() > etaMin_ && electrons->eta() < etaMax_ ) {
        if ( electrons->pt() > singleTrackPtMin_ ) accepted1 = true;
        if ( electrons->pt() > diTrackPtMin_ ) nTrackOver2ndCut++;
      }
    }
  }
  catch (const edm::Exception& e) {
    //wrong reason for exception
    if ( e.categoryCode() != edm::errors::ProductNotFound ) throw;
  }


/*
 *  Don't use candidate merger for now which is flaky
 * try
 * {
 *   iEvent.getByLabel(trackLabel_, tracks);
 * }
 *
 * catch (...) 
 * {	
 *   edm::LogError("HiggsToWW2LeptonsSkim") << "FAILED to get Track Collection. ";
 *   return false;
 * }
 *
 * if ( tracks->empty() ) {
 *   return false;
 * }
 *
 * // at least one track above a pt threshold singleTrackPtMin 
 * // and at least 2 tracks above a pt threshold diTrackPtMin
 * for( size_t c = 0; c != tracks->size(); ++ c ) {
 *   CandidateRef cref( tracks, c );
 *   if ( cref->pt() > singleTrackPtMin_ && cref->eta() > etaMin_ && cref->eta() < etaMax_ ) accepted1 = true;
 *   if ( cref->pt() > diTrackPtMin_     && cref->eta() > etaMin_ && cref->eta() < etaMax_ )  nTrackOver2ndCut++;
 * }
 *
 */




  if ( accepted1 && nTrackOver2ndCut >= 2 ) accepted = true;

  if ( accepted ) nAccepted_++;

  return accepted;

}
