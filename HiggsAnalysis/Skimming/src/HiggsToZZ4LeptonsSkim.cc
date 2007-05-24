
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

  // Pt cut on electron and muon (HLT mimic)
  singleMuonHLTPtCut = pset.getParameter<double>("singleMuonHLTPtCut");
  doubleMuonHLTPtCut = pset.getParameter<double>("doubleMuonHLTPtCut");
  singleElecHLTEtCut = pset.getParameter<double>("singleElecHLTEtCut");
  doubleElecHLTEtCut = pset.getParameter<double>("doubleElecHLTEtCut");

  // Minimum Pt for leptons for skimming
  muonMinPt          = pset.getParameter<double>("muonMinimumPt");
  elecMinEt          = pset.getParameter<double>("electronMinimumEt");

}


// Filter event
bool HiggsToZZ4LeptonsSkim::skim(edm::Event& event, const edm::EventSetup& setup) {

 //using namespace edm;
  using reco::TrackCollection;

  bool keepEvent   = false;

  bool oneMuon     = false;
  bool twoMuon     = false;
  bool oneElectron = false;
  bool twoElectron = false;
  bool fourLepton  = false;

  int nLeptons = 0;
  int nMuonAbovePt = 0;
  int nElectronAbovePt = 0;
  

  // Get the muon track collection from the event
  edm::Handle<reco::TrackCollection> muTracks;

  string rtrk = "recoTracks";
  string gmu = "globalMuons";
  try {
    event.getByLabel(rtrk, gmu, muTracks);
    
    reco::TrackCollection::const_iterator muons;
        
    // Loop over muon collections and count how many muons there are, 
    // and how many are above threshold
    for ( muons = muTracks->begin(); muons != muTracks->end(); ++muons ) {
      float pt_mu = muons->pt();
      if ( pt_mu > singleMuonHLTPtCut) {
        oneMuon = true;             
        nMuonAbovePt++; 
      } 
      else if ( pt_mu > doubleMuonHLTPtCut ) {
        nMuonAbovePt++; 
      }

      if ( pt_mu > muonMinPt) nLeptons++; 
    }  
  } 
  
  catch ( cms::Exception& ex ) {
    edm::LogError("HiggsTo4LeptonsSkim") << "Error! cannot get collection with label " 
					 << theGLBMuonLabel.label();
  }


  // 2 muon HLT equivalent trigger:
  if ( nMuonAbovePt > 1 ) twoMuon = true;


  // Get the electron track collection from the event
  edm::Handle<reco::PixelMatchGsfElectronCollection> pTracks;

  try {
    event.getByLabel(thePixelGsfELabel,pTracks);
    const reco::PixelMatchGsfElectronCollection* eTracks = pTracks.product();

    reco::PixelMatchGsfElectronCollection::const_iterator electrons;

    // Loop over electron collections and count how many muons there are, 
    // and how many are above threshold
    for ( electrons = eTracks->begin(); electrons != eTracks->end(); ++electrons ) {
      float et_e = electrons->et(); 
      if ( et_e > singleElecHLTEtCut) {
        oneElectron = true;
        nElectronAbovePt++; 
      }
      else if ( et_e > doubleElecHLTEtCut ) {
        nElectronAbovePt++; 
      }

      if ( et_e > elecMinEt) nLeptons++; 
    }
  }
  
  catch ( cms::Exception& ex ) {
    edm::LogError("HiggsTo4LeptonsSkim") << "Error! cannot get collection with label " 
					 << thePixelGsfELabel.label();
  }

  // 2 electron HLT equivalent trigger:
  if ( nElectronAbovePt > 1 ) twoElectron = true;

  
  if ( nLeptons > 3) fourLepton = true;

  // Make decision:
  if (( oneMuon || twoMuon || oneElectron || twoElectron ) && fourLepton ) keepEvent = true;
  
  return keepEvent;
}


