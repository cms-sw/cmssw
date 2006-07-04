#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackFinder.h"

/** \class GlobalMuonTrackFinder
 *  class to build Combined Tracks from standalone and tracker muon Track
 *  or build tracker and combined Tracks from standalone muon Track
 *  using GlobalMuonTrajectoryBuilder
 *
 *  $Date: 2006/07/02 03:02:17 $
 *  $Revision: 1.1 $
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

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"

using namespace std;

/// constructor
GlobalMuonTrackFinder::GlobalMuonTrackFinder(GlobalMuonTrajectoryBuilder *trajectoryBuilder) : 
theTrajectoryBuilder(trajectoryBuilder)
{
  theTkTrackLabel = "";

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

  std::vector<Trajectory> cTrajs;  
  // reconstruct the traj
  unsigned int position;
  for (position = 0; position < staMuons->size(); position++) {
      reco::TrackRef staMuon(staMuons,position);
      // build trajectory, both vector size should be 0 or 1
      std::vector<Trajectory> trajs = theTrajectoryBuilder->trajectories(&staMuon,event,eSetup);
      std::vector<reco::TrackRef*> tkTracks = theTrajectoryBuilder->chosenTrackerTrackRef();
//      if (trajs.size() >1 || tkTracks.size() >1)  something wrong;
//      if (trajs.size() != tkTracks.size() )  something wrong;

      cTrajs.push_back(trajs.front());

      reco::Muon recMuon;
// link reco Muon with standalone and tracker Track
      recMuon.setTrack(*tkTracks.front());
      recMuon.setStandAlone(staMuon);
      gMuonColl->push_back(recMuon);

  }

//      theTrackLoader->(cTrajs,event);

// put combined Track into Event
  edm::OrphanHandle<reco::TrackCollection> orphanHandleTrack = event.put(combinedTColl);

// second loop links combined Track with Muon

  position =0; 
  for (reco::MuonCollection::iterator recMuon = gMuonColl->begin(); recMuon != gMuonColl->end(); recMuon++) {
    const reco::TrackRef combinedTR(orphanHandleTrack,position); 
    recMuon->setCombined(combinedTR);
    position++;
  }
// put muon into Event
  event.put(gMuonColl);
  
}
