
/* \class HiggsTo2GammaSkim 
 *
 * Consult header file for description
 *
 * author:  Kati Lassila-Perini Helsinki Institute of Physics
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsTo2GammaSkim.h>

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Message logger
#include <FWCore/MessageLogger/interface/MessageLogger.h>

// Photons:
#include <DataFormats/EgammaCandidates/interface/Photon.h>

// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsTo2GammaSkim::HiggsTo2GammaSkim(const edm::ParameterSet& pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsTo2GammaSkim");

  // Reconstructed objects
  thePhotonLabel     = pset.getParameter<edm::InputTag>("PhotonCollectionLabel");
//  thePhotonLabel     = pset.getParameter<std::string>("PhotonCollectionLabel");

  // Minimum Pt for photons for skimming
  photon1MinPt       = pset.getParameter<double>("photon1MinimumPt");
  photon2MinPt       = pset.getParameter<double>("photon2MinimumPt"); 
  nPhotonMin         = pset.getParameter<int>("nPhotonMinimum");


  nEvents         = 0;
  nSelectedEvents = 0;

}


// Destructor
HiggsTo2GammaSkim::~HiggsTo2GammaSkim() {

  edm::LogVerbatim("HiggsTo2GammaSkim") 
  << " Number_events_read " << nEvents          
  << " Number_events_kept " << nSelectedEvents 
  << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;
}



// Filter event
bool HiggsTo2GammaSkim::filter(edm::Event& event, const edm::EventSetup& setup ) {

  nEvents++;

  using reco::PhotonCollection;

  bool keepEvent    = false;
  int  nPhotons     = 0;
  int  nHighPhotons = 0;  

  // Look at photons:

  try {
    // Get the photon collection from the event
    edm::Handle<reco::PhotonCollection> photonHandle;

    event.getByLabel(thePhotonLabel.label(),photonHandle);
    // event.getByLabel(thePhotonLabel, photonHandle);
    const reco::PhotonCollection* phoCollection = photonHandle.product();

    reco::PhotonCollection::const_iterator photons;

    // Loop over photon collections and count how many photons there are, 
    // and how many are above the thresholds

    // Question: do we need to take the reconstructed primary vertex at this point?
    // Here, I assume that the et is taken with respect to the nominal vertex (0,0,0).
    for ( photons = phoCollection->begin(); photons != phoCollection->end(); ++photons ) {
      float et_p = photons->et(); 
      if ( et_p > photon1MinPt) nPhotons++;
      if ( et_p > photon2MinPt) nHighPhotons++; 
      if ( debug ) edm::LogInfo("HiggsTo2GammaSkim") << "Photon pt " << et_p << std::endl;
    }
  }
  
  catch (...) {
    edm::LogError("HiggsTo2GammaSkim") << "Warning: cannot get collection with label " 
					   << thePhotonLabel;
  }

  
  // Make decision:
  if ( nPhotons >= nPhotonMin && nHighPhotons >= 1 ) keepEvent = true;
  if ( debug ) edm::LogInfo("HiggsTo2GammaSkim") << "n photons " << nPhotons << std::endl;

  if (keepEvent) nSelectedEvents++;

  return keepEvent;
}


