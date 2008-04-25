/**
 *  Class: GlobalCosmicMuonTrajectoryBuilder
 *
 *  $Date: 2007/12/16 13:56:15 $
 *  $Revision: 1.8 $
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 *
 **/

#include "RecoMuon/CosmicMuonProducer/interface/GlobalCosmicMuonTrajectoryBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
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
  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");
  thePropagatorName = par.getParameter<string>("Propagator");
  
}

//
// destructor
//

GlobalCosmicMuonTrajectoryBuilder::~GlobalCosmicMuonTrajectoryBuilder() {

  if (theSmoother) delete theSmoother;

}

//
// set Event
//
void GlobalCosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  event.getByLabel(theTkTrackLabel,theTrackerTracks);

  edm::Handle<std::vector<Trajectory> > handleTrackerTrajs;
  if ( event.getByLabel(theTkTrackLabel,handleTrackerTrajs) ) {
      tkTrajsAvailable = true;
      allTrackerTrajs = &*handleTrackerTrajs;   
      LogInfo("GlobalCosmicMuonTrajectoryBuilder") 
	<< "Tk Trajectories Found! " << endl;
  } else {
      LogInfo("GlobalCosmicMuonTrajectoryBuilder") 
	<< "No Tk Trajectories Found! " << endl;
      tkTrajsAvailable = false;
  }

   theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    theService->eventSetup().get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalCosmicMuonTrajectoryBuilder::trajectories(const TrackCand& muCand) {

  const std::string metname = "Muon|RecoMuon|CosmicMuon|GlobalCosmicMuonTrajectoryBuilder";
  MuonCandidate::CandidateContainer result;

  LogTrace(metname) <<"Found "<<theTrackerTracks->size()<<" tracker Tracks";
  if (theTrackerTracks->empty()) return result;

  LogTrace(metname) <<"It has "<<theTrackerTracks->front().found()<<" tk rhs";

  //at most 1 track by SingleTrackPattern
  reco::TrackRef tkTrack(theTrackerTracks,0); 
  reco::TrackRef muTrack = muCand.second;

  if ( !match(*muTrack,*tkTrack).first ) return result;

  ConstRecHitContainer muRecHits;

  if (muCand.first == 0 || !muCand.first->isValid()) { 
     muRecHits = getTransientRecHits(*muTrack);
  } else {
     muRecHits = muCand.first->recHits();
  }

  LogTrace(metname)<<"mu RecHits: "<<muRecHits.size();

  ConstRecHitContainer tkRecHits;

  if ( !tkTrajsAvailable ) {
     tkRecHits = getTransientRecHits(*tkTrack);
  } else {
     tkRecHits = allTrackerTrajs->front().recHits();
  }

  ConstRecHitContainer hits; //= tkRecHits;
  LogTrace(metname)<<"tk RecHits: "<<tkRecHits.size();

//  hits.insert(hits.end(), muRecHits.begin(), muRecHits.end());
//  stable_sort(hits.begin(), hits.end(), DecreasingGlobalY());
  sortHits(hits, muRecHits, tkRecHits);

  LogTrace(metname) << "Used RecHits after sort: "<<hits.size();
  LogTrace(metname)<<utilities()->print(hits);
  LogTrace(metname) << "== End of Used RecHits == ";

  TrajectoryStateTransform tsTrans;

  TrajectoryStateOnSurface muonState1 = tsTrans.innerStateOnSurface(*muTrack, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface tkState1 = tsTrans.innerStateOnSurface(*tkTrack, *theService->trackingGeometry(), &*theService->magneticField());

  TrajectoryStateOnSurface muonState2 = tsTrans.outerStateOnSurface(*muTrack, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface tkState2 = tsTrans.outerStateOnSurface(*tkTrack, *theService->trackingGeometry(), &*theService->magneticField());

  TrajectoryStateOnSurface firstState1 =
   ( muonState1.globalPosition().y() > tkState1.globalPosition().y() )? muonState1 : tkState1;
  TrajectoryStateOnSurface firstState2 =
   ( muonState2.globalPosition().y() > tkState2.globalPosition().y() )? muonState2 : tkState2;

  TrajectoryStateOnSurface firstState =
   ( firstState1.globalPosition().y() > firstState2.globalPosition().y() )? firstState1 : firstState2;

  if (!firstState.isValid()) return result;
  
  LogTrace(metname) <<"firstTSOS pos: "<<firstState.globalPosition()<<"mom: "<<firstState.globalMomentum();

  // begin refitting

  TrajectorySeed seed;
  vector<Trajectory> refitted = theSmoother->trajectories(seed,hits,firstState);

  if ( refitted.empty() ) refitted = theSmoother->fit(seed,hits,firstState); //FIXME

  if (refitted.empty()) {
     LogTrace(metname)<<"refit fail";
     return result;
  }

  Trajectory* myTraj = new Trajectory(refitted.front());

  std::vector<TrajectoryMeasurement> mytms = myTraj->measurements(); 
  LogTrace(metname)<<"measurements in final trajectory "<<mytms.size();
  LogTrace(metname) <<"Orignally there are "<<tkTrack->found()<<" tk rhs and "<<muTrack->found()<<" mu rhs.";

  if ( mytms.size() <= tkTrack->found() ) {
     LogTrace(metname)<<"insufficient measurements. skip... ";
     return result;
  }

  MuonCandidate* myCand = new MuonCandidate(myTraj,muTrack,tkTrack);
  result.push_back(myCand);
  LogTrace(metname)<<"final global cosmic muon: ";
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

void GlobalCosmicMuonTrajectoryBuilder::sortHits(ConstRecHitContainer& hits, ConstRecHitContainer& muonHits, ConstRecHitContainer& tkHits) {

  std::string metname = "Muon|RecoMuon|CosmicMuon|GlobalCosmicMuonTrajectoryBuilder";
  //sort hits going from higher to lower positions
  if ( tkHits.front()->globalPosition().y() < tkHits.back()->globalPosition().y() )  {//check if tk hits order same direction
    reverse(tkHits.begin(), tkHits.end());
  }

  if ( muonHits.front()->globalPosition().y() < muonHits.back()->globalPosition().y() )  {//check if tk hits order same direction
    reverse(muonHits.begin(), muonHits.end());
  }

  //separate muon hits into 2 different hemisphere
  ConstRecHitContainer::iterator middlepoint = muonHits.begin();
  bool insertInMiddle = false;

  for (ConstRecHitContainer::iterator ihit = muonHits.begin(); 
       ihit != muonHits.end() - 1; ihit++ ) {
    GlobalPoint ipos = (*ihit)->globalPosition();
    GlobalPoint nextpos = (*(ihit+1))->globalPosition();
    GlobalPoint middle((ipos.x()+nextpos.x())/2, (ipos.y()+nextpos.y())/2, (ipos.z()+nextpos.z())/2);
    LogTrace(metname)<<"ipos "<<ipos<<"nextpos"<<nextpos<<" middle "<<middle;
    if ( (middle.perp() < ipos.perp()) && (middle.perp() < nextpos.perp() ) ) {
      LogTrace(metname)<<"found middlepoint";
      middlepoint = ihit;
      insertInMiddle = true;
      break;
    }
  }

  //insert track hits in correct order
  if ( insertInMiddle ) { //if tk hits should be sandwich
    GlobalPoint jointpointpos = (*middlepoint)->globalPosition();
    LogTrace(metname)<<"jointpoint "<<jointpointpos;
    if ((tkHits.front()->globalPosition() - jointpointpos).mag() > (tkHits.back()->globalPosition() - jointpointpos).mag() ) {//check if tk hits order same direction
      reverse(tkHits.begin(), tkHits.end());
    }
    muonHits.insert(middlepoint+1, tkHits.begin(), tkHits.end());
    hits = muonHits; 
  } else { // append at one end
    if ( (tkHits.front()->globalPosition() - muonHits.back()->globalPosition()).y() < 0 ) { //insert at the end
      hits = muonHits; 
      hits.insert(hits.end(), tkHits.begin(), tkHits.end());
    } else { //insert at the beginning
      hits = tkHits;
      hits.insert(hits.end(), muonHits.begin(), muonHits.end());
    }
  }
}


TransientTrackingRecHit::ConstRecHitContainer
GlobalCosmicMuonTrajectoryBuilder::getTransientRecHits(const reco::Track& track) const {

  TransientTrackingRecHit::ConstRecHitContainer result;
  
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )
	result.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon ){
	result.push_back(theMuonRecHitBuilder->build(&**hit));
      }
  return result;
}
