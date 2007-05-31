#include <HiggsAnalysis/Skimming/interface/VBFHiggsTo2TauLJetSkim.h>

// User include files
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

// Muons:
// #include <DataFormats/TrackReco/interface/Track.h>

// Electrons:
// #include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

// Other particles ??? --> You may need to update the BuildFile as well !

// C++
// #include <algorithm>
// #include <iostream>
// #include <vector>

using namespace std;
using namespace edm;
// using namespace reco;

VBFHiggsTo2TauLJetSkim::VBFHiggsTo2TauLJetSkim(const edm::ParameterSet& pset) : HiggsAnalysisSkimType(pset) 
{
 debug = pset.getParameter<bool>("Debug");

  // Eventually, HLT objects:

  // Reconstructed objects
//   recTrackLabel      = pset.getParameter<edm::InputTag>("RecoTrackLabel");
//   theGLBMuonLabel    = pset.getParameter<edm::InputTag>("GlobalMuonCollectionLabel");
//   thePixelGsfELabel  = pset.getParameter<edm::InputTag>("ElectronCollectionLabel");

}

// --------------------------------------------------------------------------------------------

bool VBFHiggsTo2TauLJetSkim::skim(edm::Event& event, const edm::EventSetup& setup) {

  bool keepEvent = false;

  // Enter your filtering algorithm here...
  
  return keepEvent;
}
