
/* \class HiggsTo4LeptonsSkimProducer 
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 * modified by N. De Filippis - LLR - Ecole Polytechnique
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimProducer.h>
#include "DataFormats/Common/interface/Handle.h"

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
HiggsToZZ4LeptonsSkimProducer::HiggsToZZ4LeptonsSkimProducer(const edm::ParameterSet& pset) {

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

  aliasaccept="flagSkimaccept";
  produces<bool> (aliasaccept).setBranchAlias(aliasaccept);

}


// Destructor
HiggsToZZ4LeptonsSkimProducer::~HiggsToZZ4LeptonsSkimProducer() {

}

// Produce flags for event
void HiggsToZZ4LeptonsSkimProducer::produce(edm::Event& event, const edm::EventSetup& setup ) {

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

      if(muons->isGlobalMuon()){
	if ( muons->pt() > stiffMinPt) nStiffLeptons++; 
	if ( muons->pt() > softMinPt) nLeptons++; 
      }
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
  
  auto_ptr<bool> flagaccept ( new bool );
  *flagaccept=keepEvent;
  event.put(flagaccept,aliasaccept);

}

void HiggsToZZ4LeptonsSkimProducer::endJob() {
}
