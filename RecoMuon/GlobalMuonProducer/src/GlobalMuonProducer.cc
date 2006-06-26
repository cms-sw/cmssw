/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from internal seeds (muon track segments).
 *
 *
 *   $Date: 2006/06/05 08:20:49 $
 *   $Revision: 1.5 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"

#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"

// TrackFinder and Specific GLB Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

using namespace edm;
using namespace std;

/// constructor with config
GlobalMuonProducer::GlobalMuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet GLB_pSet = parameterSet.getParameter<ParameterSet>("GLBTrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<string>("MuonSeedCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder

  produces<reco::TrackCollection>();
//  produces<reco::MuonCollection>();
}
  
/// destructor
GlobalMuonProducer::~GlobalMuonProducer(){
}


/// reconstruct muons
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
  // Take the seeds container
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Reconstruct the tracks in the tracker+muon system
  std::auto_ptr<reco::TrackCollection> recTracks;
 //   = theTrackFinder->reconstruct(seeds);

  // Create a Muon Collection which holds the above infos plus the STA one
  
  
  // Load the RecMuon and the Tracks Containers in the Event
  event.put(recTracks);
}

