/**
 *  Class: GlobalCosmicMuonTrajectoryBuilder
 *
 *  $Date: 2006/09/22 18:58:17 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 *
 **/

#include "RecoMuon/CosmicMuonProducer/interface/GlobalCosmicMuonTrajectoryBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"



using namespace std;
using namespace edm;

//
// constructor
//
GlobalCosmicMuonTrajectoryBuilder::GlobalCosmicMuonTrajectoryBuilder(const edm::ParameterSet& par,
						                     const MuonServiceProxy* service) : theService(service) {
  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new MuonTrackReFitter(refitterPSet,theService);
  theTkTrackLabel = par.getParameter<string>("TkTrackCollectionLabel");
  theTTRHBuilderName = par.getParameter<string>("TrackRecHitBuilder");
  theTrackConverter = new MuonTrackConverter(par,theService);
  
}

//
// destructor
//

GlobalCosmicMuonTrajectoryBuilder::~GlobalCosmicMuonTrajectoryBuilder() {

  if (theRefitter) delete theRefitter;
  if (theTrackConverter) delete theTrackConverter;

}

//
// set Event
//
void GlobalCosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  event.getByLabel(theTkTrackLabel,theTrackerTracks);

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalCosmicMuonTrajectoryBuilder::trajectories(const reco::TrackRef& muTrack) {

  const std::string metname = "GlobalCosmicMuonTrajectoryBuilder";

  LogDebug(metname) <<"Found "<<theTrackerTracks->size()<<" tracker Tracks";
  if (theTrackerTracks->empty()) return MuonCandidate::CandidateContainer();

  LogDebug(metname) <<"It has "<<theTrackerTracks->front().found()<<" tk rhs";

  //at most 1 track by SingleTrackPattern
  reco::TrackRef tkTrack(theTrackerTracks,0); 
  MuonCandidate::CandidateContainer result;

  if ( !match(*muTrack,*tkTrack).first ) return result;

  std::vector<Trajectory> muTrajs = theTrackConverter->convert(*muTrack);
  LogDebug(metname) <<"Converted "<<muTrajs.size()<<" muon Trajectory";
  if ( muTrajs.empty() ) return result;
  ConstRecHitContainer muRecHits = muTrajs.front().recHits();
  LogDebug(metname)<<"mu RecHits: "<<muRecHits.size();

  std::vector<Trajectory> tkTrajs = theTrackConverter->convert(*tkTrack);
  LogDebug(metname) <<"Converted "<<tkTrajs.size()<<" tracker Trajectory";

  ConstRecHitContainer hits = theTrackConverter->getTransientRecHits(*tkTrack);
  LogDebug(metname)<<"tk RecHits: "<<hits.size();

  hits.insert(hits.end(), muRecHits.begin(), muRecHits.end());
  stable_sort(hits.begin(), hits.end(), DecreasingGlobalY());

  LogDebug(metname) << "Used RecHits After sort: "<<hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug(metname) << "invalid RecHit";
      continue;
    }
    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug(metname)
    << "global (" << pos.x()<<", " << pos.y()<<", " << pos.z()<<") "
    << "local: (" << (*ir)->localPosition().x()<<", " << (*ir)->localPosition().y()<<", " << (*ir)->localPosition().z()<<") "
    << "  dimension = " << (*ir)->dimension()
    << "  id " << (*ir)->det()->geographicalId().rawId();
  }
  
  TrajectoryStateOnSurface firstState;

  if (! tkTrajs.empty() ) {

    // choose the up state as first state to do backward refitting
    firstState = tkTrajs.front().firstMeasurement().updatedState();
     (tkTrajs.front().firstMeasurement().updatedState().globalPosition().y()
     > tkTrajs.front().lastMeasurement().updatedState().globalPosition().y())? 
     tkTrajs.front().firstMeasurement().updatedState() 
     : muTrajs.front().lastMeasurement().updatedState();

  }

  if (( !firstState.isValid()) || (firstState.isValid() && firstState.globalMomentum().mag() < 7.0 )) {

    // choose the up state to extrapolate to tracker
    firstState = muTrajs.front().firstMeasurement().updatedState();
     (muTrajs.front().firstMeasurement().updatedState().globalPosition().y()
     > muTrajs.front().lastMeasurement().updatedState().globalPosition().y())?
     muTrajs.front().firstMeasurement().updatedState()
     : muTrajs.front().lastMeasurement().updatedState();

   // propagate to first rechit surface from mu Trajectory

    firstState = theRefitter->propagator(oppositeToMomentum)->propagate(firstState, hits.front()->det()->surface());
 
  }
  if (!firstState.isValid()) return result;
  
  LogDebug(metname) <<"firstTSOS pos: "<<firstState.globalPosition()<<"mom: "<<firstState.globalMomentum();

  // begin refitting

  TrajectorySeed seed;
  vector<Trajectory> refitted = theRefitter->trajectories(seed,hits,firstState);

  if (refitted.empty()) {
     LogDebug(metname)<<"refit fail";
     return result;
  }

  Trajectory* myTraj = new Trajectory(refitted.front());

  std::vector<TrajectoryMeasurement> mytms = myTraj->measurements(); 
  LogDebug(metname)<<"measurements in final trajectory "<<mytms.size();
  if ( mytms.size() <= tkTrack->found() ) {
     LogDebug(metname)<<"too little measurements. skip... ";
     return result;
  }

  MuonCandidate* myCand = new MuonCandidate(myTraj,muTrack,tkTrack);
  result.push_back(myCand);

  for (std::vector<TrajectoryMeasurement>::const_iterator itm = mytms.begin();
       itm != mytms.end(); ++itm ) {
       LogDebug(metname)<<"updated pos "<<itm->updatedState().globalPosition()
                       <<"mom "<<itm->updatedState().globalMomentum();
       }
  return result;
}

std::pair<bool,double> GlobalCosmicMuonTrajectoryBuilder::match(const reco::Track& muTrack, const reco::Track& tkTrack) {

  float deltaPhi = muTrack.phi() - tkTrack.phi();
  float deltaEta = muTrack.eta() - tkTrack.eta();

  float deltaR = sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta);

  if (deltaR < 2 )
   return pair<bool,double>(true, 0);

  return pair<bool,double>(false, 0);

} 
