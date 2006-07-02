#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackFinder.h"

/** \class GlobalMuonTrackFinder
 *  class to build Combined Tracks from standalone and tracker muon Track
 *  or build tracker and combined Tracks from standalone muon Track
 *  using GlobalMuonTrajectoryBuilder
 *
 *  $Date:  $
 *  $Revision: $
 *  \author Chang Liu - Purdue University
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"

using namespace std;

/// constructor
GlobalMuonTrackFinder::GlobalMuonTrackFinder(GlobalMuonTrajectoryBuilder *trajectoryBuilder, GlobalMuonTrackMatcher* trackMatcher) : 
theTrajectoryBuilder(trajectoryBuilder),
theTrackMatcher(trackMatcher)
{
}

/// destructor
GlobalMuonTrackFinder::~GlobalMuonTrackFinder(){
//  delete theTrajectoryBuilder;
};

// Reconstruct trajectories
void GlobalMuonTrackFinder::reconstruct(const edm::Handle<reco::TrackCollection>& staMuons,
				  edm::Event &event,
				  const edm::EventSetup &eSetup) const{

  std::auto_ptr<reco::TrackCollection> combinedTColl(new reco::TrackCollection());
  std::auto_ptr<reco::MuonCollection> gMuonColl(new reco::MuonCollection());
   
  // reconstruct the traj
  for (int position = 0; position < int(staMuons->size()); position++) {

      //get tk TrackCollection from Event
      edm::Handle<reco::TrackCollection> tkTrackC;

      // get sta Track from Collection
      reco::TrackRef staTrack(staMuons, position);

      //choose one tkTrack from the Collection 
      pair<bool,reco::TrackRef> tkTrack = theTrackMatcher->match(staTrack, tkTrackC);

      if (!tkTrack.first) continue;

 //     Trajectory combinedTraj;// = theTrajectoryBuilder.build(staTrack, tkTrack);
      reco::Track combinedTrack; // = buildTrack(gtTrajectories.first);
      combinedTColl->push_back(combinedTrack);

      reco::Muon recMuon;
// link reco Muon with standalone and tracker Track
      recMuon.setTrack(tkTrack.second);
      recMuon.setStandAlone(staTrack);
      gMuonColl->push_back(recMuon);

  }

// put combined Track into Event
  edm::OrphanHandle<reco::TrackCollection> orphanHandleTrack = event.put(combinedTColl);

// second loop links combined Track with Muon

  int position =0; 
  for (reco::MuonCollection::iterator recMuon = gMuonColl->begin(); recMuon != gMuonColl->end(); recMuon++) {
    const reco::TrackRef combinedTR(orphanHandleTrack,position); 
    recMuon->setCombined(combinedTR);
    position++;
  }
// put muon into Event
  event.put(gMuonColl);
  
}
