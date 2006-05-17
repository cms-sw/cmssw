/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from internal seeds (muon track segments).
 *
 *
 *   $Date: 2006/04/13 15:30:02 $
 *   $Revision: 1.1 $
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
#include "RecoMuon/GlobalTrackFinder/interface/GlobalTrajectoryBuilder.h"


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

using namespace edm;

/// constructor with config
GlobalMuonProducer::GlobalMuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet GLB_pSet = parameterSet.getParameter<ParameterSet>("GLBTrajBuilder");

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new GlobalMuonTrajectoryBuilder(GLB_pSet));


  produces<reco::TrackCollection>();  //FIXME fancy-name  
}
  
/// destructor
GlobalMuonProducer::~GlobalMuonProducer(){
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
  // ##### FIXME fancy-names  #####

  // Take the seeds container
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel("MuonSeedsForGlobal",seeds);

  // Reconstruct 
  std::auto_ptr<reco::TrackCollection> recMuons
    = theTrackFinder->reconstruct(seeds,eventSetup);

  // the best would be 
  //   = theTrackFinder->reconstruct(seeds);
  
  // Load the RecMuon Container in the Event
  event.put(recMuons);
}

