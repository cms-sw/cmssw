
/* \class HiggsToXexampleSkim
 *
 * A template to use to create different skim
 *
 * author:  Dominique Fortin - UC Riverside
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToXexampleSkim.h>

// User include files
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

// Muons:
#include <DataFormats/TrackReco/interface/Track.h>

// Electrons:
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

// Other particles ??? --> You may need to update the BuildFile as well !

// C++
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;


// Constructor
HiggsToXexampleSkim::HiggsToXexampleSkim(const edm::ParameterSet& pset) : HiggsAnalysisSkimType(pset) {

  // Local Debug flag
  debug              = pset.getParameter<bool>("DebugHiggsToXexampleSkim");

  // Eventually, HLT objects:

  // Reconstructed objects
  recTrackLabel      = pset.getParameter<edm::InputTag>("RecoTrackLabel");
  theGLBMuonLabel    = pset.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
  thePixelGsfELabel  = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

  // Load in some criteria here:
  dummyCut = pset.getParameter<double>("dummyCut");

}


// Filter event
bool HiggsToXexampleSkim::skim(edm::Event& event, const edm::EventSetup& setup) {

  bool keepEvent = false;

  // Enter your filtering algorithm here...
  
  return keepEvent;
}


