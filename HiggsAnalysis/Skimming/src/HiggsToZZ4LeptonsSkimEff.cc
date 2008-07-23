
/* \class HiggsTo4LeptonsSkimEff 
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimEff.h>

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Muons:
#include <DataFormats/TrackReco/interface/Track.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToZZ4LeptonsSkimEff::HiggsToZZ4LeptonsSkimEff(const edm::ParameterSet& pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsSkim");

  // Reconstructed objects
  recTrackLabel      = pset.getParameter<edm::InputTag>("RecoTrackLabel");
  theGLBMuonLabel    = pset.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
  thePixelGsfELabel  = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

  // Minimum Pt for leptons for skimming
  // Minimum Pt for leptons for skimming
  stiffMinPt         = pset.getParameter<double>("stiffMinimumPt");
  softMinPt          = pset.getParameter<double>("softMinimumPt");
  nStiffLeptonMin    = pset.getParameter<int>("nStiffLeptonMinimum");
  nLeptonMin         = pset.getParameter<int>("nLeptonMinimum");

  nEvents   = 0;
  nSelFourE = nSelFourM = nSelTwoETwoM = nSelFourL = nSelTau = 0;
  nFourE    = nFourM    = nTwoETwoM    = nFourL    = nTau    = 0;

}


// Destructor
HiggsToZZ4LeptonsSkimEff::~HiggsToZZ4LeptonsSkimEff() {

  std::cout << "Number of events read " << nEvents << std::endl;
  std::cout << "*** Efficiency for the various subsamples *** " <<  endl;

  std::cout << "Four leptons: " 
  << " pres "    << nFourL         
  << " kept "    << nSelFourL
  << " eff  "    << ((double)nSelFourL)/((double) nFourL + 0.0001) << std::endl;
  std::cout << "Four muons:   "
  << " pres "    << nFourM         
  << " kept "    << nSelFourM
  << " eff  "    << ((double)nSelFourM)/((double) nFourM + 0.0001) << std::endl;
  std::cout << "Four elecs:   "
  << " pres "    << nFourE         
  << " kept "    << nSelFourE
  << " eff  "    << ((double)nSelFourE)/((double) nFourE + 0.0001) << std::endl;
  std::cout << "2 elec 2 mu:  "
  << " pres "    << nTwoETwoM   
  << " kept "    << nSelTwoETwoM
  << " eff  "    << ((double)nSelTwoETwoM)/((double) nTwoETwoM + 0.0001) << std::endl;
  std::cout << "with taus:    "
  << " pres "    << nTau     
  << " kept "    << nSelTau
  << " eff  "    << ((double)nSelTau)/((double) nTau + 0.0001) << std::endl;

}



// Filter event
void HiggsToZZ4LeptonsSkimEff::analyze(const edm::Event& event, const edm::EventSetup& setup ) {

  nEvents++;

  using reco::TrackCollection;

  bool keepEvent   = false;

  // First, pre-selection:
  int nMuon = 0;
  int nElec = 0;
  int nTau  = 0;

  bool isFourE = false;
  bool isFourM = false;
  bool isTwoETwoM = false;
  bool isFourL = false;
  bool isTau = false;

  // get gen particle candidates
  edm::Handle<CandidateCollection> genCandidates;
  event.getByLabel("genParticleCandidates", genCandidates);

  for ( CandidateCollection::const_iterator mcIter=genCandidates->begin(); mcIter!=genCandidates->end(); ++mcIter ) {

    // Muons:
    if ( mcIter->pdgId() == 13 || mcIter->pdgId() == -13) {
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
       // In fiducial volume:
        if ( mcIter->eta() > -2.4 && mcIter->eta() < 2.4 ) nMuon++;
      }
    }
    // Electrons:
    if ( mcIter->pdgId() == 11 || mcIter->pdgId() == -11) {
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
        // In fiducial volume:
        if ( mcIter->eta() > -2.5 && mcIter->eta() < 2.5 ) nElec++;
      }
    }
    // Taus:
    if ( mcIter->pdgId() == 15 || mcIter->pdgId() == -15) {
      // Mother is a Z
      if ( mcIter->mother()->pdgId() == 23 ) {
        // In fiducial volume:
        if ( mcIter->eta() > -2.5 && mcIter->eta() < 2.5 ) nTau++;
      }
    }

  }
   
    if (nElec > 3) {
      isFourE = true;
      nFourE++;
    }
    if (nMuon > 3) {
      isFourM = true;
      nFourM++;
    }
    if (nMuon > 1 && nElec > 1) {
      isTwoETwoM = true;
      nTwoETwoM++;
    }
    if ( isFourE || isFourM || isTwoETwoM ) {
      isFourL = true;
      nFourL++;
    }
    if (nTau > 1) {
      isTau = true;
      nTau++;
    }

  if ( isFourL ) {
    keepEvent = true;
  } else {
    return;
  }


  int  nStiffLeptons = 0;
  int  nLeptons      = 0;

  // First look at muons:

  // Get the muon track collection from the event
  edm::Handle<reco::TrackCollection> muTracks;
  event.getByLabel(theGLBMuonLabel.label(), muTracks);
 
  if ( muTracks.isValid() ) {   
    reco::TrackCollection::const_iterator muons;
        
    // Loop over muon collections and count how many muons there are, 
    // and how many are above threshold
    for ( muons = muTracks->begin(); muons != muTracks->end(); ++muons ) {
      float pt_mu =  muons->pt();
      if ( pt_mu > stiffMinPt ) nStiffLeptons++; 
      if ( pt_mu > softMinPt ) nLeptons++; 
    }  
  } 
  

  // Now look at electrons:

  // Get the electron track collection from the event
  edm::Handle<reco::PixelMatchGsfElectronCollection> pTracks;
  event.getByLabel(thePixelGsfELabel.label(),pTracks);

  if ( pTracks.isValid() ) { 

    const reco::PixelMatchGsfElectronCollection* eTracks = pTracks.product();

    reco::PixelMatchGsfElectronCollection::const_iterator electrons;

    // Loop over electron collections and count how many muons there are, 
    // and how many are above threshold
    for ( electrons = eTracks->begin(); electrons != eTracks->end(); ++electrons ) {
      float pt_e = electrons->pt(); 
      if ( pt_e > stiffMinPt ) nStiffLeptons++; 
      if ( pt_e > softMinPt ) nLeptons++; 
    }
  }

  
  // Make decision:
  if ( nStiffLeptons >= nStiffLeptonMin && nLeptons >= nLeptonMin) {
    keepEvent = true;
  } else {
    keepEvent = false;
  }

  if ( keepEvent ) {
    if (isFourE)    nSelFourE++;
    if (isFourM)    nSelFourM++;
    if (isTwoETwoM) nSelTwoETwoM++;
    if (isFourL)    nSelFourL++;
    if (isTau)      nSelTau++;
  }

}
