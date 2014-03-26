#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of cosmic muons and beam-halo muons
 *
 *
 *  \author Chang Liu  - Purdue Univeristy
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

/* Collaborating Class Header */
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"

#include <algorithm>

using namespace edm;
using namespace std;

CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const edm::ParameterSet& par, const MuonServiceProxy* service, edm::ConsumesCollector& iC) : theService(service) { 

  thePropagatorName = par.getParameter<string>("Propagator");

  bool enableDTMeasurement = par.getParameter<bool>("EnableDTMeasurement");
  bool enableCSCMeasurement = par.getParameter<bool>("EnableCSCMeasurement");
  bool enableRPCMeasurement = par.getParameter<bool>("EnableRPCMeasurement");

//  if(enableDTMeasurement)
  InputTag DTRecSegmentLabel = par.getParameter<InputTag>("DTRecSegmentLabel");

//  if(enableCSCMeasurement)
  InputTag CSCRecSegmentLabel = par.getParameter<InputTag>("CSCRecSegmentLabel");

//  if(enableRPCMeasurement)
  InputTag RPCRecSegmentLabel = par.getParameter<InputTag>("RPCRecSegmentLabel");
  theLayerMeasurements= new MuonDetLayerMeasurements(DTRecSegmentLabel,
                                                     CSCRecSegmentLabel,
                                                     RPCRecSegmentLabel,
						     iC,
						     enableDTMeasurement,
						     enableCSCMeasurement,
						     enableRPCMeasurement);

  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  
  theNavigation = 0; // new DirectMuonNavigation(theService->detLayerGeometry());
  theUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet, insideOut);

  theBestMeasurementFinder = new MuonBestMeasurementFinder();

  ParameterSet muonBackwardUpdatorPSet = par.getParameter<ParameterSet>("BackwardMuonTrajectoryUpdatorParameters");

  theBKUpdator = new MuonTrajectoryUpdator(muonBackwardUpdatorPSet, outsideIn);

  theTraversingMuonFlag = par.getParameter<bool>("BuildTraversingMuon");

  theStrict1LegFlag = par.getParameter<bool>("Strict1Leg");

  ParameterSet smootherPSet = par.getParameter<ParameterSet>("MuonSmootherParameters");

  theNavigationPSet = par.getParameter<ParameterSet>("MuonNavigationParameters");

  theSmoother = new CosmicMuonSmoother(smootherPSet,theService);

  theNTraversing = 0;
  theNSuccess = 0;
  theCacheId_DG = 0;
  category_ = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

}


CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {

  LogTrace(category_)<< "CosmicMuonTrajectoryBuilder dtor called";
  if (theUpdator) delete theUpdator;
  if (theBKUpdator) delete theBKUpdator;
  if (theLayerMeasurements) delete theLayerMeasurements;
  if (theSmoother) delete theSmoother;
  if (theNavigation) delete theNavigation; 
  delete theBestMeasurementFinder;

  LogTrace(category_)<< "CosmicMuonTrajectoryBuilder Traversing: "<<theNSuccess<<"/"<<theNTraversing;

}


void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);

  // DetLayer Geometry
  unsigned long long newCacheId_DG = theService->eventSetup().get<MuonRecoGeometryRecord>().cacheIdentifier();
  if ( newCacheId_DG != theCacheId_DG ) {
    LogTrace(category_) << "Muon Reco Geometry changed!";
    theCacheId_DG = newCacheId_DG;
    if (theNavigation) delete theNavigation;
    theNavigation = new DirectMuonNavigation(theService->detLayerGeometry(), theNavigationPSet);
  }


}


MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed) {

  vector<Trajectory*> trajL = vector<Trajectory*>();
  
  MuonPatternRecoDumper debug;

  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = trajectoryStateTransform::transientState(ptsd1,&bp,&*theService->magneticField());
  LogTrace(category_) << "Seed: mom "<<lastTsos.globalMomentum()
      			  <<"pos: " <<lastTsos.globalPosition();
  LogTrace(category_)  << "Seed: mom eta "<<lastTsos.globalMomentum().eta()
                          <<"pos eta: " <<lastTsos.globalPosition().eta();
  
  bool beamhaloFlag =  ( (did.subdetId() == MuonSubdetId::CSC) && fabs(lastTsos.globalMomentum().eta()) > 4.0);

  vector<const DetLayer*> navLayers;

  if (did.subdetId() == MuonSubdetId::DT) {
    //DT
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  } 
  else if (beamhaloFlag || (theTraversingMuonFlag && theStrict1LegFlag)) {
    //CSC
    navLayers = navigation()->compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum);
  } else {
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  }

  LogTrace(category_) <<"found "<<navLayers.size()<<" compatible DetLayers for the Seed";

  if (navLayers.empty()) return trajL;
  
  vector<DetWithState> detsWithStates;
  LogTrace(category_) << "Compatible layers: ";
  for ( vector<const DetLayer*>::const_iterator layer = navLayers.begin();
        layer != navLayers.end(); layer++) {
    LogTrace(category_) << debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId()) 
                        << debug.dumpLayer(*layer);
  }

  detsWithStates = navLayers.front()->compatibleDets(lastTsos, *propagator(), *(updator()->estimator()));
  LogTrace(category_) << "Number of compatible dets: " << detsWithStates.size() << endl;

  if ( !detsWithStates.empty() ) {
    // get the updated TSOS
    if ( detsWithStates.front().second.isValid() ) {
      LogTrace(category_) << "New starting TSOS is on det: " << endl;
      LogTrace(category_) << debug.dumpMuonId(detsWithStates.front().first->geographicalId())
                        << debug.dumpLayer(navLayers.front());
      lastTsos = detsWithStates.front().second;
      LogTrace(category_) << "Seed after extrapolation: mom " << lastTsos.globalMomentum()
                          << "pos: " << lastTsos.globalPosition();
    }
  }
  detsWithStates.clear();
  if ( !lastTsos.isValid() ) return trajL;

  TrajectoryStateOnSurface secondLast = lastTsos;

  lastTsos.rescaleError(10.0);

  Trajectory* theTraj = new Trajectory(seed,alongMomentum);

  navLayers.clear();

  if (fabs(lastTsos.globalMomentum().eta()) < 1.0) {
    //DT
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  } else if (beamhaloFlag || (theTraversingMuonFlag && theStrict1LegFlag)) {
    //CSC
    navLayers = navigation()->compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum);
  } else {
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  }


  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;
  MuonTransientTrackingRecHit::MuonRecHitContainer allUnusedHits;
  vector<TrajectoryMeasurement> measL;

  LogTrace(category_) << "Begin forward fit " << navLayers.size();

  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin(); rnxtlayer!= navLayers.end(); ++rnxtlayer) {
     LogTrace(category_) << "new layer ";
     measL.clear();
     LogTrace(category_) << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId())
                         << debug.dumpLayer(*rnxtlayer);
     LogTrace(category_) << "from lastTsos " << lastTsos.globalMomentum()<< " at " <<lastTsos.globalPosition();
 
     measL = findBestMeasurements(*rnxtlayer, lastTsos, propagator(), (updator()->estimator()));

     if ( measL.empty() &&  (fabs(theService->magneticField()->inTesla(GlobalPoint(0,0,0)).z()) < 0.01) && (theService->propagator("StraightLinePropagator").isValid() ) )  {
       LogTrace(category_) << "try straight line propagator ";
       measL = findBestMeasurements(*rnxtlayer, lastTsos, &*theService->propagator("StraightLinePropagator"), (updator()->estimator()));
     }
     if ( measL.empty() ) continue;

     for (vector<TrajectoryMeasurement>::const_iterator theMeas = measL.begin(); theMeas != measL.end(); ++theMeas) {
       pair<bool,TrajectoryStateOnSurface> result = updator()->update((&*theMeas), *theTraj, propagator());

       if (result.first ) {
         LogTrace(category_) << "update ok ";
         incrementChamberCounters((*rnxtlayer), DTChamberUsedBack, CSCChamberUsedBack, RPCChamberUsedBack, TotalChamberUsedBack);
         secondLast = lastTsos;
         if ( (!theTraj->empty()) && result.second.isValid() ) {
           lastTsos = result.second;
           LogTrace(category_) << "get new lastTsos here " << lastTsos.globalMomentum() << " at " << lastTsos.globalPosition();
         }  else if ((theMeas)->predictedState().isValid()) lastTsos = (theMeas)->predictedState();
       }
     }
  } 
  measL.clear();
  while (!theTraj->empty()) {
    theTraj->pop();
  }

  if (!theTraj->isValid() || TotalChamberUsedBack < 2 || (DTChamberUsedBack+CSCChamberUsedBack) == 0 || !lastTsos.isValid()) {
    delete theTraj;
    return trajL;
  }
  delete theTraj;


  // if got good trajectory, then do backward refitting
  DTChamberUsedBack = 0;
  CSCChamberUsedBack = 0;
  RPCChamberUsedBack = 0;
  TotalChamberUsedBack = 0;

  Trajectory myTraj(seed, oppositeToMomentum);

  // set starting navigation direction for MuonTrajectoryUpdator

  GlobalPoint lastPos = lastTsos.globalPosition();
  GlobalPoint secondLastPos = secondLast.globalPosition();
  GlobalVector momDir = secondLastPos - lastPos;

  if ( lastPos.basicVector().dot(momDir.basicVector()) > 0 ) { 
//      LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction changed to insideOut";
      theBKUpdator->setFitDirection(insideOut);
    } else theBKUpdator->setFitDirection(outsideIn);

  if (fabs(lastTsos.globalMomentum().eta()) < 1.0) {
    //DT
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);
  } else if (beamhaloFlag || (theTraversingMuonFlag && theStrict1LegFlag)) {
    //CSC
    std::reverse(navLayers.begin(), navLayers.end());
  } else {
    navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);  
  } 

  LogTrace(category_) << "Begin backward refitting, with " << navLayers.size() << " layers" << endl;

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     measL.clear();

     measL = findBestMeasurements(*rnxtlayer, lastTsos, propagator(), (backwardUpdator()->estimator()));

     if ( measL.empty() ) {
         MuonTransientTrackingRecHit::MuonRecHitContainer tmpHits = theLayerMeasurements->recHits(*rnxtlayer);
         for (MuonRecHitContainer::const_iterator ihit = tmpHits.begin();
                                           ihit != tmpHits.end(); ++ihit ) {
         allUnusedHits.push_back(*ihit);
         } 
         continue;
      }

     for (vector<TrajectoryMeasurement>::const_iterator theMeas = measL.begin(); theMeas != measL.end(); ++theMeas) {

      // if the part change, we need to reconsider the fit direction
       if (rnxtlayer != navLayers.begin()) {

         vector<const DetLayer*>::const_iterator lastlayer = rnxtlayer;
         lastlayer--;

         if ( (*rnxtlayer)->location() != (*lastlayer)->location() ) {

           lastPos = lastTsos.globalPosition();
           GlobalPoint thisPos = (theMeas)->predictedState().globalPosition();
           GlobalVector momDir = thisPos - lastPos;
//         LogTrace("CosmicMuonTrajectoryBuilder")<<"momDir "<<momDir;
 
           if ( momDir.mag() > 0.01 ) { //if lastTsos is on the surface, no need
             if ( thisPos.basicVector().dot(momDir.basicVector()) > 0 ) {
               theBKUpdator->setFitDirection(insideOut);
             } else theBKUpdator->setFitDirection(outsideIn);
           }
         }
         if ( ((*lastlayer)->location() == GeomDetEnumerators::endcap) && 
              ((*rnxtlayer)->location() == GeomDetEnumerators::endcap) && 
              (lastTsos.globalPosition().z() * (theMeas)->predictedState().globalPosition().z() < 0)  ) {
           theBKUpdator->setFitDirection(insideOut);
         }
     }

//       if (theBKUpdator->fitDirection() == insideOut) 
//          LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction insideOut";
//       else LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction outsideIn";
         pair<bool,TrajectoryStateOnSurface> bkresult
              = backwardUpdator()->update((&*theMeas), myTraj, propagator());

         if (bkresult.first ) {

              incrementChamberCounters((*rnxtlayer), DTChamberUsedBack, CSCChamberUsedBack, RPCChamberUsedBack, TotalChamberUsedBack);

            if ( theTraversingMuonFlag ) {

              MuonRecHitContainer tmpUnusedHits = unusedHits(*rnxtlayer,*theMeas);
              allUnusedHits.insert(allUnusedHits.end(),tmpUnusedHits.begin(),tmpUnusedHits.end());
            }
            if ( (!myTraj.empty()) && bkresult.second.isValid() ) 
               lastTsos = bkresult.second;
            else if ((theMeas)->predictedState().isValid()) 
               lastTsos = (theMeas)->predictedState();
          }
       }
  }

  for ( vector<Trajectory*>::iterator t = trajL.begin(); t != trajL.end(); ++t ) delete *t;

  trajL.clear();

  if (( !myTraj.isValid() ) || ( myTraj.empty() ) || ( (selfDuplicate(myTraj)) )|| TotalChamberUsedBack < 2 || (DTChamberUsedBack+CSCChamberUsedBack) < 1) {
      return trajL;
  }

  if ( theTraversingMuonFlag && ( allUnusedHits.size() >= 2 )) { 
//      LogTrace(category_)<<utilities()->print(allUnusedHits);
      LogTrace(category_)<<"Building trajectory in second hemisphere...";
      buildSecondHalf(myTraj);
      // check if traversing trajectory has hits in both hemispheres

      if ( theStrict1LegFlag && !utilities()->isTraversing(myTraj) ) {trajL.clear(); return trajL;}
  } else if (theStrict1LegFlag && theTraversingMuonFlag) {trajL.clear(); return trajL;} 

  LogTrace(category_) <<" traj ok ";

//     getDirectionByTime(myTraj);
  if (beamhaloFlag) estimateDirection(myTraj);
  if ( myTraj.empty() ) return trajL;

  // try to smooth it 
  vector<Trajectory> smoothed = theSmoother->trajectories(myTraj); 

  if ( !smoothed.empty() && smoothed.front().foundHits()> 3 )  {  
    LogTrace(category_) <<" Smoothed successfully."; 
    myTraj = smoothed.front(); 
  } 
  else {  
    LogTrace(category_) <<" Smooth failed."; 
  } 

  LogTrace(category_) <<"first "<< myTraj.firstMeasurement().updatedState()
                      <<"\n last "<<myTraj.lastMeasurement().updatedState();
  if ( myTraj.direction() == alongMomentum ) LogTrace(category_)<<"alongMomentum";
  else if (myTraj.direction() == oppositeToMomentum ) LogTrace(category_)<<"oppositeMomentum";
  else LogTrace(category_)<<"anyDirection";

  if (!beamhaloFlag) {
      if ( myTraj.lastMeasurement().updatedState().globalMomentum().y() > 0 ) {
          LogTrace(category_)<<"flip trajectory ";
          flipTrajectory(myTraj);
      }

      if ( ( myTraj.direction() == alongMomentum && 
           (myTraj.firstMeasurement().updatedState().globalPosition().y() 
           < myTraj.lastMeasurement().updatedState().globalPosition().y()))
        || (myTraj.direction() == oppositeToMomentum && 
           (myTraj.firstMeasurement().updatedState().globalPosition().y() 
           > myTraj.lastMeasurement().updatedState().globalPosition().y())) ) {
           LogTrace(category_)<<"reverse propagation direction";
          reverseTrajectoryPropagationDirection(myTraj); 
      }
  }
//  getDirectionByTime(myTraj);
  if ( !myTraj.isValid() ) return trajL;

  // check direction agree with position!
  PropagationDirection dir = myTraj.direction();
  GlobalVector dirFromPos = myTraj.measurements().back().recHit()->globalPosition() - myTraj.measurements().front().recHit()->globalPosition();

  if ( theStrict1LegFlag && !utilities()->isTraversing(myTraj) ) {trajL.clear(); return trajL;}

  LogTrace(category_)<< "last hit " <<myTraj.measurements().back().recHit()->globalPosition()<<endl;
  LogTrace(category_)<< "first hit " <<myTraj.measurements().front().recHit()->globalPosition()<<endl;

  LogTrace(category_)<< "last tsos " <<myTraj.measurements().back().updatedState().globalPosition()<<" mom "<<myTraj.measurements().back().updatedState().globalMomentum()<<endl;
  LogTrace(category_)<< "first tsos " <<myTraj.measurements().front().updatedState().globalPosition()<<" mom "<<myTraj.measurements().front().updatedState().globalMomentum()<<endl;

  PropagationDirection propDir =
      ( dirFromPos.basicVector().dot(myTraj.firstMeasurement().updatedState().globalMomentum().basicVector()) > 0) ? alongMomentum : oppositeToMomentum;
  LogTrace(category_)<<" dir "<<dir <<" propDir "<<propDir<<endl;

  LogTrace(category_)<<"chi2 "<<myTraj.chiSquared() <<endl;

  if (dir != propDir ) {
      LogTrace(category_)<< "reverse propagation direction ";
      reverseTrajectoryPropagationDirection(myTraj);
  }
  if ( myTraj.empty() ) return trajL;

  trajL.push_back(new Trajectory(myTraj));
  navLayers.clear();
  return trajL;
}


//
//
//
MuonTransientTrackingRecHit::MuonRecHitContainer 
CosmicMuonTrajectoryBuilder::unusedHits(const DetLayer* layer, const TrajectoryMeasurement& meas) const {

  MuonTransientTrackingRecHit::MuonRecHitContainer tmpHits = theLayerMeasurements->recHits(layer);
  MuonRecHitContainer result;
  for (MuonRecHitContainer::const_iterator ihit = tmpHits.begin();
       ihit != tmpHits.end(); ++ihit ) {
       if ((*ihit)->geographicalId() != meas.recHit()->geographicalId() ){ 
         result.push_back(*ihit);
         LogTrace(category_) << "Unused hit: " << (*ihit)->globalPosition() << endl;
    }
  }

  return result;

}


//
// continue to build a trajectory starting from given trajectory state
//
void CosmicMuonTrajectoryBuilder::build(const TrajectoryStateOnSurface& ts, 
                                        const NavigationDirection& startingDir,
                                        Trajectory& traj) {

  if ( !ts.isValid() ) return;

  const FreeTrajectoryState* fts = ts.freeState();
  if ( !fts ) return;

  vector<const DetLayer*> navLayers;

  if (fabs(fts->momentum().basicVector().eta()) < 1.0) {
    //DT
    if (fts->position().basicVector().dot(fts->momentum().basicVector())>0){
       navLayers = navigation()->compatibleLayers((*fts), alongMomentum);
      } else {
        navLayers = navigation()->compatibleLayers((*fts), oppositeToMomentum);
    }

  } else if (theTraversingMuonFlag && theStrict1LegFlag) {
    //CSC
      if (fts->position().basicVector().dot(fts->momentum().basicVector())>0){
           navLayers = navigation()->compatibleEndcapLayers((*fts), alongMomentum);
         } else {
            navLayers = navigation()->compatibleEndcapLayers((*fts), oppositeToMomentum);
          }
  } else {

    if (fts->position().basicVector().dot(fts->momentum().basicVector())>0){
       navLayers = navigation()->compatibleLayers((*fts), alongMomentum);
      } else {
        navLayers = navigation()->compatibleLayers((*fts), oppositeToMomentum);
   }

}

  if (navLayers.empty()) return;

  theBKUpdator->setFitDirection(startingDir);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;

  TrajectoryStateOnSurface lastTsos = 
      (traj.lastMeasurement().updatedState().globalPosition().y() <
      traj.firstMeasurement().updatedState().globalPosition().y()) ? 
      propagatorAlong()->propagate((*fts),navLayers.front()->surface()) : propagatorOpposite()->propagate((*fts),navLayers.front()->surface());

  if ( !lastTsos.isValid() ) { 
    LogTrace(category_)<<"propagation failed from fts to inner cylinder";
    return;
  }
  LogTrace(category_)<<"tsos  "<<lastTsos.globalPosition();
  lastTsos.rescaleError(10.);
  vector<TrajectoryMeasurement> measL;
  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {

    measL.clear();
    measL = findBestMeasurements(*rnxtlayer, lastTsos, propagator(), (backwardUpdator()->estimator()));

    if ( measL.empty() ) continue;

    for (vector<TrajectoryMeasurement>::const_iterator theMeas = measL.begin(); theMeas != measL.end(); ++theMeas) {

      pair<bool,TrajectoryStateOnSurface> bkresult
             = backwardUpdator()->update((&*theMeas), traj, propagator());
      if (bkresult.first ) {
        LogTrace(category_)<<"update ok : "<<(theMeas)->recHit()->globalPosition() ;

        incrementChamberCounters((*rnxtlayer), DTChamberUsedBack, CSCChamberUsedBack, RPCChamberUsedBack, TotalChamberUsedBack);

        if ( (!traj.empty()) && bkresult.second.isValid() ) 
          lastTsos = bkresult.second;
        else if ((theMeas)->predictedState().isValid()) 
          lastTsos = (theMeas)->predictedState();
      }
    }
  }
  navLayers.clear();
  updator()->makeFirstTime();
  backwardUpdator()->makeFirstTime();

  measL.clear();

  return;

}


//
// build trajectory in second hemisphere with pattern
// recognition starting from an intermediate state
//
void CosmicMuonTrajectoryBuilder::buildSecondHalf(Trajectory& traj) {

  if ( (traj.firstMeasurement().recHit()->globalPosition().perp()
      < traj.lastMeasurement().recHit()->globalPosition().perp()) ) {
    LogTrace(category_)<<"inside-out: reverseTrajectory"; 
    reverseTrajectory(traj);
  }
  if (traj.empty()) return;
  TrajectoryStateOnSurface tsos = traj.lastMeasurement().updatedState();
  if ( !tsos.isValid() ) tsos = traj.lastMeasurement().predictedState();
 LogTrace(category_)<<"last tsos on traj: pos: "<< tsos.globalPosition()<<" mom: "<< tsos.globalMomentum();
  if ( !tsos.isValid() ) {
    LogTrace(category_)<<"last tsos on traj invalid";
    return;
  }

  build(intermediateState(tsos),insideOut,traj);

  return;

}


//
//
//
TrajectoryStateOnSurface CosmicMuonTrajectoryBuilder::intermediateState(const TrajectoryStateOnSurface& tsos) const {

  PerpendicularBoundPlaneBuilder planeBuilder;
  GlobalPoint pos(0.0, 0.0, 0.0);
  BoundPlane* SteppingPlane = planeBuilder(pos,tsos.globalDirection());

  TrajectoryStateOnSurface predTsos = propagator()->propagate(tsos, *SteppingPlane);
  if ( predTsos.isValid() )
  LogTrace(category_)<<"intermediateState: a intermediate state: pos: "<<predTsos.globalPosition() << "mom: " << predTsos.globalMomentum();

  return predTsos;

}


//
//
//
void CosmicMuonTrajectoryBuilder::selectHits(MuonTransientTrackingRecHit::MuonRecHitContainer& hits) const {

  if ( hits.size() < 2 ) return;

  MuonRecHitContainer tmp;
  vector<bool> keep(hits.size(),true);
  int i(0);
  int j(0);

  for (MuonRecHitContainer::const_iterator ihit = hits.begin();
       ihit != hits.end(); ++ihit ) {
    if ( !keep[i] ) { i++; continue; };
    j = i + 1;
    for (MuonRecHitContainer::const_iterator ihit2 = ihit + 1;
         ihit2 != hits.end(); ++ihit2 ) {
         if ( !keep[j] ) { j++; continue; }
         if ((*ihit)->geographicalId() == (*ihit2)->geographicalId() ) {
           if ( (*ihit)->dimension() > (*ihit2)->dimension() ) {
              keep[j] = false;
           } else if ( (*ihit)->dimension() < (*ihit2)->dimension() ) {
              keep[i] = false;
           } else  {
           if ( (*ihit)->transientHits().size()>(*ihit2)->transientHits().size() ) { 
              keep[j] = false;
           } else if ( (*ihit)->transientHits().size()<(*ihit2)->transientHits().size() ) {
              keep[i] = false;
           } 
            else if ( (*ihit)->degreesOfFreedom() != 0 && (*ihit2)->degreesOfFreedom() != 0)  {
            if (((*ihit)->chi2()/(*ihit)->degreesOfFreedom()) > ((*ihit2)->chi2()/(*ihit)->degreesOfFreedom())) keep[i] = false;
            else keep[j] = false;
           }
          }
         } // if same geomid 
      j++;
    }
    i++;
  }

  i = 0;
  for (MuonRecHitContainer::const_iterator ihit = hits.begin();
       ihit != hits.end(); ++ihit ) {
     if (keep[i] ) tmp.push_back(*ihit);
     i++;
  }

  hits.clear();
  hits.swap(tmp);

  return;

}


//
//
//
bool CosmicMuonTrajectoryBuilder::selfDuplicate(const Trajectory& traj) const {

  TransientTrackingRecHit::ConstRecHitContainer const & hits = traj.recHits();

  if (traj.empty()) return true;

  bool result = false;
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() )  continue;
    for (ConstRecHitContainer::const_iterator ir2 = ir+1; ir2 != hits.end(); ir2++ ) {
      if ( !(*ir2)->isValid() )  continue;
      if ( (*ir) == (*ir2) ) result = true;
    }
  }

  return result;

}


//
// reverse a trajectory without refitting
// this can only be used for cosmic muons that come from above
//
void CosmicMuonTrajectoryBuilder::reverseTrajectory(Trajectory& traj) const {

  PropagationDirection newDir = (traj.firstMeasurement().recHit()->globalPosition().y()
      < traj.lastMeasurement().recHit()->globalPosition().y())
  ? oppositeToMomentum : alongMomentum;
  Trajectory newTraj(traj.seed(), newDir);
  
  /* does not work in gcc4.8?)
  std::vector<TrajectoryMeasurement> & meas = traj.measurements();
  for (auto itm = meas.rbegin(); itm != meas.rend(); ++itm ) {
    newTraj.push(std::move(*itm));
  }
  traj = std::move(newTraj);
  */

  std::vector<TrajectoryMeasurement> const & meas = traj.measurements();
  for (auto itm = meas.rbegin(); itm != meas.rend(); ++itm ) {
    newTraj.push(*itm);
  }
  traj = newTraj;


}


//
// reverse a trajectory momentum direction and then refit
//
void CosmicMuonTrajectoryBuilder::flipTrajectory(Trajectory& traj) const {

  TrajectoryStateOnSurface lastTSOS = traj.lastMeasurement().updatedState();
  if ( !lastTSOS.isValid() ) {
    LogTrace(category_) << "Error: last TrajectoryState invalid.";
  }  
  TransientTrackingRecHit::ConstRecHitContainer hits = traj.recHits();
  std::reverse(hits.begin(), hits.end());

  LogTrace(category_) << "last tsos before flipping "<<lastTSOS;
  utilities()->reverseDirection(lastTSOS,&*theService->magneticField());
  LogTrace(category_) << "last tsos after flipping "<<lastTSOS;

  vector<Trajectory> refittedback = theSmoother->fit(traj.seed(),hits,lastTSOS);
  if ( refittedback.empty() ) {
    LogTrace(category_) <<"flipTrajectory fail. "<<endl;
    return;
  }
  LogTrace(category_) <<"flipTrajectory: first "<< refittedback.front().firstMeasurement().updatedState()
                       <<"\nflipTrajectory: last "<<refittedback.front().lastMeasurement().updatedState();

  traj = refittedback.front();

  return;

}


//
//
//
void CosmicMuonTrajectoryBuilder::reverseTrajectoryPropagationDirection(Trajectory& traj) const {

  if ( traj.direction() == anyDirection ) return;
  PropagationDirection newDir = (traj.direction() == alongMomentum)? oppositeToMomentum : alongMomentum;
  Trajectory newTraj(traj.seed(), newDir);
  const std::vector<TrajectoryMeasurement>& meas = traj.measurements();

  for (std::vector<TrajectoryMeasurement>::const_iterator itm = meas.begin(); itm != meas.end(); ++itm) {
    newTraj.push(*itm);
  }

  while (!traj.empty()) {
    traj.pop();
  }

  traj = newTraj;

}


//
// guess the direction by normalized chi2
//
void CosmicMuonTrajectoryBuilder::estimateDirection(Trajectory& traj) const {

  TransientTrackingRecHit::ConstRecHitContainer hits = traj.recHits();

  TrajectoryStateOnSurface firstTSOS = traj.firstMeasurement().updatedState();

  TrajectoryStateOnSurface lastTSOS = traj.lastMeasurement().updatedState();

  if ( !firstTSOS.isValid() || !lastTSOS.isValid() ) return;

  LogTrace(category_) <<"Two ends of the traj "<<firstTSOS.globalPosition()
                      <<", "<<lastTSOS.globalPosition();

  LogTrace(category_) <<"Their mom: "<<firstTSOS.globalMomentum()
                      <<", "<<lastTSOS.globalMomentum();

  LogTrace(category_) <<"Their mom eta: "<<firstTSOS.globalMomentum().eta()
                      <<", "<<lastTSOS.globalMomentum().eta();

  // momentum eta can be used to estimate direction
  // the beam-halo muon seems enter with a larger |eta|

  if ( fabs(firstTSOS.globalMomentum().eta()) > fabs(lastTSOS.globalMomentum().eta()) ) {

    vector<Trajectory> refitted = theSmoother->trajectories(traj.seed(),hits,firstTSOS);
    if ( !refitted.empty() ) traj = refitted.front();

  } else {
    std::reverse(hits.begin(), hits.end());
    utilities()->reverseDirection(lastTSOS,&*theService->magneticField());
    vector<Trajectory> refittedback = theSmoother->trajectories(traj.seed(),hits,lastTSOS);
    if ( !refittedback.empty() ) traj = refittedback.front();

  }

  return;

}


//
// get direction from timing information of rechits and segments
//
void CosmicMuonTrajectoryBuilder::getDirectionByTime(Trajectory& traj) const {

  TransientTrackingRecHit::ConstRecHitContainer hits = traj.recHits();
  LogTrace(category_) << "getDirectionByTime"<<endl;
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogTrace(category_) << "invalid RecHit"<<endl;
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogTrace(category_)
    << "pos" << pos
    << "radius " << pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector()<<endl;

    if ((*ir)->det()->geographicalId().det() == 2 && (*ir)->det()->subDetector() == 6) { 
//      const CSCRecHit2D* iCSC = dynamic_cast<const CSCRecHit2D*>(&**ir);
//      if (iCSC) LogTrace(category_)<<"csc from cast tpeak "<<iCSC->tpeak(); 
      CSCRecHit2DCollection::range thisrange = cschits_->get(CSCDetId((*ir)->geographicalId()));
      for (CSCRecHit2DCollection::const_iterator rechit = thisrange.first; rechit!=thisrange.second;++rechit) {
         if ((*rechit).isValid()) LogTrace(category_)<<"csc from collection tpeak "<<(*rechit).tpeak();
      }
    }
    if ((*ir)->det()->geographicalId().det() == 2 && (*ir)->det()->subDetector() == 7) {
//      const DTRecHit1D* iDT = dynamic_cast<const DTRecHit1D*>(&**ir);
//      if (iDT) LogTrace(category_)<<"dt digitime "<<iDT->digiTime();
      DTRecHitCollection::range thisrange = dthits_->get(DTLayerId((*ir)->geographicalId()));
      for (DTRecHitCollection::const_iterator rechit = thisrange.first; rechit!=thisrange.second;++rechit) {
         if ((*rechit).isValid()) LogTrace(category_)<<"dt from collection digitime "<<(*rechit).digiTime();
      }
    }
  }

  return;

}


//
//
//
std::vector<TrajectoryMeasurement>
CosmicMuonTrajectoryBuilder::findBestMeasurements(const DetLayer* layer,
                                                  const TrajectoryStateOnSurface& tsos, 
                                                  const Propagator* propagator, 
                                                  const MeasurementEstimator* estimator) {

  std::vector<TrajectoryMeasurement> result;
  std::vector<TrajectoryMeasurement> measurements;

  if ( layer->hasGroups() ) {
    std::vector<TrajectoryMeasurementGroup> measurementGroups =
      theLayerMeasurements->groupedMeasurements(layer, tsos, *propagator, *estimator);

    for (std::vector<TrajectoryMeasurementGroup>::const_iterator tmGroupItr = measurementGroups.begin();
        tmGroupItr != measurementGroups.end(); ++tmGroupItr) {
    
      measurements = tmGroupItr->measurements();
      const TrajectoryMeasurement* bestMeasurement 
        = theBestMeasurementFinder->findBestMeasurement(measurements, propagator);
      
      if (bestMeasurement) result.push_back(*bestMeasurement);
    }
  } 
  else {
    measurements = theLayerMeasurements->measurements(layer, tsos, *propagator, *estimator);
    const TrajectoryMeasurement* bestMeasurement 
      = theBestMeasurementFinder->findBestMeasurement(measurements, propagator);

    if (bestMeasurement) result.push_back(*bestMeasurement);
  }
  measurements.clear();

  return result;

}


//
//
//
void CosmicMuonTrajectoryBuilder::incrementChamberCounters(const DetLayer* layer,
                                                           int& dtChambers, 
                                                           int& cscChambers, 
                                                           int& rpcChambers, 
                                                           int& totalChambers) {

  if (layer->subDetector()==GeomDetEnumerators::DT) dtChambers++; 
  else if (layer->subDetector()==GeomDetEnumerators::CSC) cscChambers++; 
  else if (layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcChambers++; 
  totalChambers++;

}


//
//
//
double CosmicMuonTrajectoryBuilder::t0(const DTRecSegment4D* dtseg) const {

   if ( (dtseg == 0) || (!dtseg->hasPhi()) ) return 0;
   // timing information
   double result = 0;
   if ( dtseg->phiSegment() == 0 ) return 0; 
   int phiHits = dtseg->phiSegment()->specificRecHits().size();
   LogTrace(category_) << "phiHits " << phiHits;
   if ( phiHits > 5 ) {
     if(dtseg->phiSegment()->ist0Valid()) result = dtseg->phiSegment()->t0();
     if (dtseg->phiSegment()->ist0Valid()){
       LogTrace(category_) << " Phi t0: " << dtseg->phiSegment()->t0() << " hits: " << phiHits;
     } else {
       LogTrace(category_) << " Phi t0 is invalid: " << dtseg->phiSegment()->t0() << " hits: " << phiHits;
     }
   }

   return result;

}


//
//
//
PropagationDirection CosmicMuonTrajectoryBuilder::checkDirectionByT0(const DTRecSegment4D* dtseg1, 
                                                                     const DTRecSegment4D* dtseg2) const {

   LogTrace(category_) << "comparing dtseg: " << dtseg1 << " " << dtseg2 << endl;
   if (dtseg1 == dtseg2 || t0(dtseg1) == t0(dtseg2)) return anyDirection; 

   PropagationDirection result =
    ( t0(dtseg1) < t0(dtseg2) ) ? alongMomentum : oppositeToMomentum;

   return result;

}
