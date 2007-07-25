
/* \class HiggsTo4LeptonsSkim 
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkim.h>

// User include files
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

// Muons:
#include <DataFormats/TrackReco/interface/Track.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

// C++
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToZZ4LeptonsSkim::HiggsToZZ4LeptonsSkim(const edm::ParameterSet& pset) : HiggsAnalysisSkimType(pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsSkim");

  // Eventually, HLT objects:

  // Reconstructed objects
  recTrackLabel      = pset.getParameter<edm::InputTag>("RecoTrackLabel");
  theGLBMuonLabel    = pset.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
  thePixelGsfELabel  = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

  // Minimum Pt for leptons for skimming
  muonMinPt          = pset.getParameter<double>("muonMinimumPt");
  elecMinEt          = pset.getParameter<double>("electronMinimumEt");
  nLeptonMin         = pset.getParameter<int>("nLeptonMinimum");

}


// Filter event
bool HiggsToZZ4LeptonsSkim::skim(edm::Event& event, const edm::EventSetup& setup ) {

  //using namespace edm;
  using reco::TrackCollection;

  bool keepEvent   = false;
  int  nLeptons    = 0;
  

  // First look at muons:

  try {
  // Get the muon track collection from the event
    edm::Handle<reco::TrackCollection> muTracks;
    event.getByLabel(theGLBMuonLabel.label(), muTracks);
    
    reco::TrackCollection::const_iterator muons;
        
    // Loop over muon collections and count how many muons there are, 
    // and how many are above threshold
    for ( muons = muTracks->begin(); muons != muTracks->end(); ++muons ) {
      if ( muons->pt() > muonMinPt) nLeptons++; 
    }  
  } 
  
  catch (...) {
    edm::LogError("HiggsToZZ4LeptonsSkim") << "Warning: cannot get collection with label " 
			    		   << theGLBMuonLabel.label();
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
      float et_e = electrons->et(); 
      if ( et_e > elecMinEt) nLeptons++; 
    }
  }
  
  catch (...) {
    edm::LogError("HiggsToZZ4LeptonsSkim") << "Warning: cannot get collection with label " 
					   << thePixelGsfELabel.label();
  }

  
  // Make decision:
  if ( nLeptons >= nLeptonMin) keepEvent = true;
  
  return keepEvent;
}


