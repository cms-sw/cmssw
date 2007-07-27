#include <HiggsAnalysis/Skimming/interface/VBFHiggsTo2TauLJetSkim.h>

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

VBFHiggsTo2TauLJetSkim::VBFHiggsTo2TauLJetSkim(const edm::ParameterSet& pset) {

 debug = pset.getParameter<bool>("Debug");

  // Reconstructed objects
//   recTrackLabel      = pset.getParameter<edm::InputTag>("RecoTrackLabel");
//   theGLBMuonLabel    = pset.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
//   thePixelGsfELabel  = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

  nEvents         = 0;
  nSelectedEvents = 0;

}


VBFHiggsTo2TauLJetSkim::~VBFHiggsTo2TauLJetSkim() {

  edm::LogVerbatim("VBFHiggsTo2TauLJetSkim")
  << " Number_events_read " << nEvents
  << " Number_events_kept " << nSelectedEvents
  << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;
}


bool VBFHiggsTo2TauLJetSkim::filter(edm::Event& event, const edm::EventSetup& setup) {

  nEvents++;

  bool keepEvent = false;

  // Enter your filtering algorithm here...
  

  if (keepEvent) nSelectedEvents++;

  return keepEvent;
}
