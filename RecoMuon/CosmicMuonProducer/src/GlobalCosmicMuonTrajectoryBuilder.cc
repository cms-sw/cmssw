/**
 *  Class: GlobalCosmicMuonTrajectoryBuilder
 *
 *  $Date: 2007/03/21 18:23:29 $
 *  $Revision: 1.6 $
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
  ParameterSet smootherPSet = par.getParameter<ParameterSet>("SmootherParameters");
  theSmoother = new CosmicMuonSmoother(smootherPSet,theService);
  theTkTrackLabel = par.getParameter<string>("TkTrackCollectionLabel");
  theTTRHBuilderName = par.getParameter<string>("TrackRecHitBuilder");
  thePropagatorName = par.getParameter<string>("Propagator");

  ParameterSet transformerPSet = par.getParameter<ParameterSet>("TransformerParameters");
  theTrackTransformer = new TrackTransformer(transformerPSet);
  
}

//
// destructor
//

GlobalCosmicMuonTrajectoryBuilder::~GlobalCosmicMuonTrajectoryBuilder() {

  if (theSmoother) delete theSmoother;
  if (theTrackTransformer) delete theTrackTransformer;

}

//
// set Event
//
void GlobalCosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  event.getByLabel(theTkTrackLabel,theTrackerTracks);

  edm::Handle<std::vector<Trajectory> > handleTrackerTrajs;
  try
    {
      event.getByLabel(theTkTrackLabel,handleTrackerTrajs);
      tkTrajsAvailable = true;
      allTrackerTrajs = &*handleTrackerTrajs;   
      
      LogInfo("GlobalCosmicMuonTrajectoryBuilder") 
	<< "Tk Trajectories Found! " << endl;
    }
  catch (...)
    {
      LogInfo("GlobalCosmicMuonTrajectoryBuilder") 
	<< "No Tk Trajectories Found! " << endl;
      tkTrajsAvailable = false;
    }

  theTrackTransformer->setServices(theService->eventSetup());

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalCosmicMuonTrajectoryBuilder::trajectories(const TrackCand& muCand) {

  const std::string metname = "GlobalCosmicMuonTrajectoryBuilder";

  LogTrace(metname) <<"Found "<<theTrackerTracks->size()<<" tracker Tracks";
  if (theTrackerTracks->empty()) return MuonCandidate::CandidateContainer();

  LogTrace(metname) <<"It has "<<theTrackerTracks->front().found()<<" tk rhs";

  //at most 1 track by SingleTrackPattern
  reco::TrackRef tkTrack(theTrackerTracks,0); 
  MuonCandidate::CandidateContainer result;
  reco::TrackRef muTrack = muCand.second;

  if ( !match(*muTrack,*tkTrack).first ) return result;
  std::vector<Trajectory> muTrajs;
  if (muCand.first == 0) { 
     muTrajs = theTrackTransformer->transform(*muTrack);
  }
  else if ( !muCand.first->isValid() ) {
     muTrajs = theTrackTransformer->transform(*muTrack);
  }
  else muTrajs.push_back(*muCand.first);

  LogTrace(metname) <<"There're "<<muTrajs.size()<<" muon Trajectory";
  if ( muTrajs.empty() ) return result;
  ConstRecHitContainer muRecHits = muTrajs.front().recHits();
  LogTrace(metname)<<"mu RecHits: "<<muRecHits.size();

  std::vector<Trajectory> tkTrajs;
  if (!tkTrajsAvailable) {
     tkTrajs = theTrackTransformer->transform(*tkTrack);
  } 
  else tkTrajs = *allTrackerTrajs;
  LogTrace(metname) <<"Converted "<<tkTrajs.size()<<" tracker Trajectory";

  ConstRecHitContainer hits = tkTrajs.front().recHits();
  LogTrace(metname)<<"tk RecHits: "<<hits.size();

  hits.insert(hits.end(), muRecHits.begin(), muRecHits.end());
  stable_sort(hits.begin(), hits.end(), DecreasingGlobalY());

  LogTrace(metname) << "Used RecHits After sort: "<<hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogTrace(metname) << "invalid RecHit";
      continue;
    }
    const GlobalPoint& pos = (*ir)->globalPosition();
    LogTrace(metname)
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

    firstState = theService->propagator(thePropagatorName)->propagate(firstState, hits.front()->det()->surface());
 
  }
  if (!firstState.isValid()) return result;
  
  LogTrace(metname) <<"firstTSOS pos: "<<firstState.globalPosition()<<"mom: "<<firstState.globalMomentum();

  // begin refitting

  TrajectorySeed seed;
  vector<Trajectory> refitted = theSmoother->trajectories(seed,hits,firstState);

  if (refitted.empty()) {
     LogTrace(metname)<<"refit fail";
     return result;
  }

  Trajectory* myTraj = new Trajectory(refitted.front());

  std::vector<TrajectoryMeasurement> mytms = myTraj->measurements(); 
  LogTrace(metname)<<"measurements in final trajectory "<<mytms.size();
  if ( mytms.size() <= tkTrack->found() ) {
     LogTrace(metname)<<"too little measurements. skip... ";
     return result;
  }

  MuonCandidate* myCand = new MuonCandidate(myTraj,muTrack,tkTrack);
  result.push_back(myCand);

  for (std::vector<TrajectoryMeasurement>::const_iterator itm = mytms.begin();
       itm != mytms.end(); ++itm ) {
       LogTrace(metname)<<"updated pos "<<itm->updatedState().globalPosition()
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
