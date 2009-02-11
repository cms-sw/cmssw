
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
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// Muons
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToZZ4LeptonsSkim::HiggsToZZ4LeptonsSkim(const edm::ParameterSet& pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsSkim");

  // Reconstructed objects
  theMuonLabel       = pset.getParameter<edm::InputTag>("MuonCollectionLabel");
  theGsfELabel       = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

  // Minimum Pt for leptons for skimming
  stiffMinPt         = pset.getParameter<double>("stiffMinimumPt");
  softMinPt          = pset.getParameter<double>("softMinimumPt");
  nStiffLeptonMin    = pset.getParameter<int>("nStiffLeptonMinimum");
  nLeptonMin         = pset.getParameter<int>("nLeptonMinimum");

  nEvents         = 0;
  nSelectedEvents = 0;

}


// Destructor
HiggsToZZ4LeptonsSkim::~HiggsToZZ4LeptonsSkim() {

  std::cout << "HiggsToZZ4LeptonsSkim: \n" 
  << " N_events_HLTread= " << nEvents          
  << " N_events_Skimkept= " << nSelectedEvents 
  << "     RelEfficiency4lFilter= " << double(nSelectedEvents)/double(nEvents) << std::endl;
}



// Filter event
bool HiggsToZZ4LeptonsSkim::filter(edm::Event& event, const edm::EventSetup& setup ) {

  nEvents++;

  using reco::MuonCollection;

  bool keepEvent   = false;
  int  nStiffLeptons    = 0;
  int  nLeptons    = 0;
  

  // First look at muons:

  // Get the muon collection from the event
  edm::Handle<reco::MuonCollection> mus;
  event.getByLabel(theMuonLabel.label(), mus);

  if ( mus.isValid() ) {  
  
    reco::MuonCollection::const_iterator muons;
        
    // Loop over muon collections and count how many muons there are, 
    // and how many are above threshold
    for ( muons = mus->begin(); muons != mus->end(); ++muons ) {
      if ( muons->pt() > stiffMinPt) nStiffLeptons++; 
      if ( muons->pt() > softMinPt) nLeptons++; 
    }  
  } 
  
  // Now look at electrons:

  // Get the electron track collection from the event
  edm::Handle<reco::GsfElectronCollection> eles;
  event.getByLabel(theGsfELabel.label(),eles);

  if ( eles.isValid() ) {  

    reco::GsfElectronCollection::const_iterator electrons;

    // Loop over electron collections and count how many muons there are, 
    // and how many are above threshold
    for ( electrons = eles->begin(); electrons != eles->end(); ++electrons ) {
      float pt_e = electrons->pt(); 
      if ( pt_e > stiffMinPt) nStiffLeptons++; 
      if ( pt_e > softMinPt) nLeptons++; 
    }
  }

  // Make decision:
  if ( nStiffLeptons >= nStiffLeptonMin && nLeptons >= nLeptonMin) keepEvent = true;

  if (keepEvent) nSelectedEvents++;

  return keepEvent;
}


