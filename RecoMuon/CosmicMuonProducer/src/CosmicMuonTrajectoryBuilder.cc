#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of cosmic muons and beam-halo muons
 *
 *
 *  $Date: 2007/08/15 16:41:10 $
 *  $Revision: 1.27 $
 *  \author Chang Liu  - Purdue Univeristy
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

/* Collaborating Class Header */
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

#include <algorithm>

using namespace edm;
using namespace std;

CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const edm::ParameterSet& par, const MuonServiceProxy*service):theService(service) { 

  thePropagatorName = par.getParameter<string>("Propagator");

  bool enableDTMeasurement = par.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  bool enableCSCMeasurement = par.getUntrackedParameter<bool>("EnableCSCMeasurement",true);
  bool enableRPCMeasurement = par.getUntrackedParameter<bool>("EnableRPCMeasurement",true);

//  if(enableDTMeasurement)
    string DTRecSegmentLabel = par.getUntrackedParameter<string>("DTRecSegmentLabel", "dt4DSegments");

//  if(enableCSCMeasurement)
    string CSCRecSegmentLabel = par.getUntrackedParameter<string>("CSCRecSegmentLabel","cscSegments");

//  if(enableRPCMeasurement)
    string RPCRecSegmentLabel = par.getUntrackedParameter<string>("CSCRecSegmentLabel","rpcRecHits");


  theLayerMeasurements= new MuonDetLayerMeasurements(enableDTMeasurement,
						     enableCSCMeasurement,
						     enableRPCMeasurement,
                                                     DTRecSegmentLabel,
                                                     CSCRecSegmentLabel,
                                                     RPCRecSegmentLabel);

  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  
  theUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet, insideOut);

  ParameterSet muonBackwardUpdatorPSet = par.getParameter<ParameterSet>("BackwardMuonTrajectoryUpdatorParameters");

  theBKUpdator = new MuonTrajectoryUpdator(muonBackwardUpdatorPSet, outsideIn);

  theTraversingMuonFlag = par.getUntrackedParameter<bool>("BuildTraversingMuon",true);

  ParameterSet smootherPSet = par.getParameter<ParameterSet>("MuonSmootherParameters");

  theSmoother = new CosmicMuonSmoother(smootherPSet,theService);

  theNTraversing = 0;
  theNSuccess = 0;

}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  LogTrace(metname)<< "CosmicMuonTrajectoryBuilder dtor called";
  if (theUpdator) delete theUpdator;
  if (theBKUpdator) delete theBKUpdator;
  if (theLayerMeasurements) delete theLayerMeasurements;
  if (theSmoother) delete theSmoother;

  LogTrace(metname)<< "CosmicMuonTrajectoryBuilder Traversing: "<<theNSuccess<<"/"<<theNTraversing;

}

void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);

}

MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";
  vector<Trajectory*> trajL;
  TrajectoryStateTransform tsTransform;
  MuonPatternRecoDumper debug;

  DirectMuonNavigation navigation((theService->detLayerGeometry()));
  MuonBestMeasurementFinder measFinder;
 
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = tsTransform.transientState(ptsd1,&bp,&*theService->magneticField());
      LogTrace(metname) << "Trajectory State on Surface of Seed";
      LogTrace(metname)<<"mom: "<<lastTsos.globalMomentum();
      LogTrace(metname)<<"pos: " <<lastTsos.globalPosition();
      LogTrace(metname)<<"eta: "<<lastTsos.globalMomentum().eta();
  
  bool beamhaloFlag =  (fabs(lastTsos.globalMomentum().eta()) > 4.5);

  vector<const DetLayer*> navLayers;
  if ( beamhaloFlag ) { //skip barrel layers for BeamHalo 
     navLayers = navigation.compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum);
  } else navLayers = navigation.compatibleLayers(*(lastTsos.freeState()), alongMomentum);


  LogTrace(metname)<<"found "<<navLayers.size()<<" compatible DetLayers for the Seed";
  if (navLayers.empty()) return trajL;
  
  vector<DetWithState> detsWithStates;
  LogTrace(metname) <<"Compatible layers:";
  for( vector<const DetLayer*>::const_iterator layer = navLayers.begin();
       layer != navLayers.end(); layer++){
    LogTrace(metname)<< debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId()) 
                     << debug.dumpLayer(*layer);
  }

  detsWithStates = navLayers.front()->compatibleDets(lastTsos, *propagator(), *(updator()->estimator()));
  LogTrace(metname)<<"Number of compatible dets: "<<detsWithStates.size()<<endl;

  if( !detsWithStates.empty() ){
    // get the updated TSOS
    if ( detsWithStates.front().second.isValid() ) {
      LogTrace(metname)<<"New starting TSOS is on det: "<<endl;
      LogTrace(metname) << debug.dumpMuonId(detsWithStates.front().first->geographicalId())
                        << debug.dumpLayer(navLayers.front());
      LogTrace(metname) << "Trajectory State on Surface after extrapolation";
      lastTsos = detsWithStates.front().second;
      LogTrace(metname)<<"mom: "<<lastTsos.globalMomentum();
      LogTrace(metname)<<"pos: " << lastTsos.globalPosition();
    }
  }

  if ( !lastTsos.isValid() ) return trajL;

  TrajectoryStateOnSurface secondLast = lastTsos;

  lastTsos.rescaleError(10.0);

  Trajectory* theTraj = new Trajectory(seed,alongMomentum);

  if ( beamhaloFlag ) {
     navLayers = navigation.compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum);
  } else navLayers = navigation.compatibleLayers(*(lastTsos.freeState()), alongMomentum);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;
  MuonTransientTrackingRecHit::MuonRecHitContainer allUnusedHits;

  LogTrace(metname)<<"Begin forward fit";
  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin(); rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(updator()->estimator()));

     LogTrace(metname)<<"There're "<<measL.size()<<" measurements in DetLayer "
     << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId());

     if ( measL.empty() ) continue;

     TrajectoryMeasurement* theMeas=measFinder.findBestMeasurement(measL,propagator());

     if ( theMeas ) {

        pair<bool,TrajectoryStateOnSurface> result
            = updator()->update(theMeas, *theTraj, propagator());

        if (result.first ) {
          LogTrace(metname)<<"update ok ";
 
          if((*rnxtlayer)-> subDetector() == GeomDetEnumerators::DT) DTChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::CSC) CSCChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCBarrel || (*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCEndcap) RPCChamberUsedBack++;
          TotalChamberUsedBack++;

          secondLast = lastTsos;
          if ( (!theTraj->empty()) && result.second.isValid() ) 
             lastTsos = result.second;
          else if (theMeas->predictedState().isValid()) lastTsos = theMeas->predictedState();
        }
      }
  } 

  if (!theTraj->isValid() || TotalChamberUsedBack < 2 || (DTChamberUsedBack+CSCChamberUsedBack) == 0) {
    delete theTraj;
    return trajL;
  }

  if ( !lastTsos.isValid() ) {
    delete theTraj;
    return trajL;
  }

  delete theTraj;

  //if got good trajectory, then do backward refitting
  DTChamberUsedBack = 0;
  CSCChamberUsedBack = 0;
  RPCChamberUsedBack = 0;
  TotalChamberUsedBack = 0;

  Trajectory* myTraj = new Trajectory(seed, oppositeToMomentum);

  // set starting navigation direction for MuonTrajectoryUpdator

  GlobalPoint lastPos = lastTsos.globalPosition();
  GlobalPoint secondLastPos = secondLast.globalPosition();
  GlobalVector momDir(secondLastPos.x()-lastPos.x(),
                      secondLastPos.y()-lastPos.y(),
                      secondLastPos.z()-lastPos.z());
//  LogTrace("CosmicMuonTrajectoryBuilder")<<"lastTsos"<<lastPos;
//  LogTrace("CosmicMuonTrajectoryBuilder")<<"secondLast"<<secondLastPos;
//  LogTrace("CosmicMuonTrajectoryBuilder")<<"momDir"<<momDir;
  if ( lastPos.x() * momDir.x()
      +lastPos.y() * momDir.y()
      +lastPos.z() * momDir.z() > 0 ){
//      LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction changed to insideOut";
      theBKUpdator->setFitDirection(insideOut);
    } else theBKUpdator->setFitDirection(outsideIn);

  if ( beamhaloFlag ) { 
     std::reverse(navLayers.begin(), navLayers.end());
  } else navLayers = navigation.compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);

  LogTrace(metname)<<"Begin backward refitting";

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(backwardUpdator()->estimator()));
     LogTrace(metname)<<"There're "<<measL.size()<<" measurements in DetLayer "
     << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId()); 

     if ( measL.empty() ) continue;

     TrajectoryMeasurement* theMeas=measFinder.findBestMeasurement(measL,propagator());

     if ( theMeas ) {
      // if the part change, we need to reconsider the fit direction

       if (rnxtlayer != navLayers.begin()) {
         vector<const DetLayer*>::const_iterator lastlayer = rnxtlayer;
         lastlayer--;

         if((*rnxtlayer)->location() != (*lastlayer)->location() ) {

            lastPos = lastTsos.globalPosition();
            GlobalPoint thisPos = theMeas->predictedState().globalPosition();
            GlobalVector momDir(thisPos.x()-lastPos.x(),
                                thisPos.y()-lastPos.y(),
                                thisPos.z()-lastPos.z());
//          LogTrace("CosmicMuonTrajectoryBuilder")<<"momDir "<<momDir;

            if ( momDir.mag() > 0.01 ) { //if lastTsos is on the surface, no need
              if ( thisPos.x() * momDir.x() 
                  +thisPos.y() * momDir.y()
                  +thisPos.z() * momDir.z() > 0 ){
                   theBKUpdator->setFitDirection(insideOut);
                } else theBKUpdator->setFitDirection(outsideIn);
            }
          }
         if ( ((*lastlayer)->location() == GeomDetEnumerators::endcap) && 
              ((*rnxtlayer)->location() == GeomDetEnumerators::endcap) && 
              (lastTsos.globalPosition().z() * theMeas->predictedState().globalPosition().z() < 0)  ) {
                  theBKUpdator->setFitDirection(insideOut);
          }

       }
//       if (theBKUpdator->fitDirection() == insideOut) 
//          LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction insideOut";
//       else LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction outsideIn";

       pair<bool,TrajectoryStateOnSurface> bkresult
            = backwardUpdator()->update(theMeas, *myTraj, propagator());

       if (bkresult.first ) {
          if((*rnxtlayer)-> subDetector() == GeomDetEnumerators::DT) DTChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::CSC) CSCChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCBarrel || (*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCEndcap) RPCChamberUsedBack++;
          TotalChamberUsedBack++;
          if ( theTraversingMuonFlag ) {
            MuonRecHitContainer tmpUnusedHits = unusedHits(*rnxtlayer,*theMeas);
            allUnusedHits.insert(allUnusedHits.end(),tmpUnusedHits.begin(),tmpUnusedHits.end());
          }
          if ( (!myTraj->empty()) && bkresult.second.isValid() ) 
             lastTsos = bkresult.second;
          else if (theMeas->predictedState().isValid()) 
             lastTsos = theMeas->predictedState();
        }
    }
  }

  TransientTrackingRecHit::ConstRecHitContainer hits = myTraj->recHits();
  unsigned int nhits = hits.size(); //for debug, remove me later...

//  LogTrace(metname) << "Used RecHits before building second half: "<<hits.size();
//  print(hits);
//  LogTrace(metname) << "== End of Used RecHits == ";

  LogTrace(metname)<<"all unused RecHits: "<<allUnusedHits.size();

  if ( theTraversingMuonFlag && ( allUnusedHits.size() >= 2 ) && 
     ( ( myTraj->lastLayer()->location() == GeomDetEnumerators::barrel ) ||
       ( myTraj->firstMeasurement().layer()->location() == GeomDetEnumerators::barrel ) ) ) {
      theNTraversing++;

      LogTrace(metname)<<utilities()->print(allUnusedHits);
   //   LogTrace(metname)<<"== End of Unused RecHits ==";
//      selectHits(allUnusedHits);
//      LogTrace(metname)<<"all unused RecHits after selection: "<<allUnusedHits.size();
//      print(allUnusedHits);
//      LogTrace(metname)<<"== End of Unused RecHits ==";

      LogTrace(metname)<<"Building trajectory in second hemisphere...";

 //     explore(*myTraj, allUnusedHits);
      buildSecondHalf(*myTraj);

      hits = myTraj->recHits();
      LogTrace(metname) << "After explore: Used RecHits: "<<hits.size();
      LogTrace(metname)<<utilities()->print(hits);
      LogTrace(metname) << "== End of Used RecHits == ";

      if ( hits.size() > nhits + 2 ) theNSuccess++;
      else LogTrace(metname) << "building on second hemisphere failed. ";
  }

  if (myTraj->isValid() && (!selfDuplicate(*myTraj)) && TotalChamberUsedBack >= 2 && (DTChamberUsedBack+CSCChamberUsedBack) > 0){
     LogTrace(metname) <<" traj ok ";

     if (beamhaloFlag) estimateDirection(*myTraj);

     for ( vector<Trajectory*>::iterator t = trajL.begin(); t != trajL.end(); ++t ) delete *t;
     trajL.clear();

     vector<Trajectory> smoothed = theSmoother->trajectories(*myTraj);
     if ( !smoothed.empty() )  {
       LogTrace(metname) <<" Smoothed successfully.";

       delete myTraj;
       Trajectory* smthed = new Trajectory(smoothed.front());
       trajL.push_back(smthed);
     }
     else {
       LogTrace(metname) <<" Smoothing failed.";
       trajL.push_back(myTraj);
     }
  }
  LogTrace(metname) <<" trajL ok "<<trajL.size();

  return trajL;
}

MuonTransientTrackingRecHit::MuonRecHitContainer CosmicMuonTrajectoryBuilder::unusedHits(const DetLayer* layer, const TrajectoryMeasurement& meas) const {
  MuonTransientTrackingRecHit::MuonRecHitContainer tmpHits = theLayerMeasurements->recHits(layer);
  MuonRecHitContainer result;
  for (MuonRecHitContainer::const_iterator ihit = tmpHits.begin();
       ihit != tmpHits.end(); ++ihit ) {
       if ((*ihit)->geographicalId() != meas.recHit()->geographicalId() ) 
         result.push_back(*ihit);
  }
  return result;

}

//
// build trajectory in the second hemisphere by picking up rechits.
// currently not used, for study only
//
void CosmicMuonTrajectoryBuilder::explore(Trajectory& traj, MuonRecHitContainer& hits) {

  //C.L.:
  //the method determine which side of the trajectory the unused hits located,
  //choose the end of Trajectory as startingTSOS which is closer to hits
  //in previous step, we know that
  //rechits in trajectory and unused rechits should have same 
  //(inside-out/outside-in) direction.
  //To combine, we must have trajectory outside-in and hits inside-out
  //Thus
  //if inside-out, reverse traj
  //if outside-out, reverse unused rechits 

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  bool trajInsideOut = (traj.firstMeasurement().recHit()->globalPosition().perp()
      < traj.lastMeasurement().recHit()->globalPosition().perp()); 

  bool hitsInsideOut = (hits.front()->globalPosition().perp()
      < hits.back()->globalPosition().perp()); 

  theBKUpdator->setFitDirection(insideOut);

  if ( trajInsideOut && hitsInsideOut ) {
    LogTrace(metname)<<"inside-out: reverseTrajectory"; 
    reverseTrajectory(traj);
    updateTrajectory(traj,hits); 

  } else if ( (!trajInsideOut) && (!hitsInsideOut)) {
    //both outside-in 
    //fit with reversed hits
    LogTrace(metname)<<"outside-in: reverse hits";
    std::reverse(hits.begin(), hits.end()); 
    updateTrajectory(traj,hits);
  
  } else {
    LogTrace(metname)<<"Error: traj and hits have different directions"; //FIXME
  } 
  return;
}


//
// continue to build a trajectory starting from given fts
//
void CosmicMuonTrajectoryBuilder::build(const TrajectoryStateOnSurface& ts, 
                                        const NavigationDirection& startingDir,
                                        Trajectory& traj) {

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  if ( !ts.isValid() ) return;

  DirectMuonNavigation navigation((theService->detLayerGeometry())); 
  MuonBestMeasurementFinder measFinder;
  MuonPatternRecoDumper debug;

  theBKUpdator->setFitDirection(startingDir);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;

  vector<const DetLayer*> navLayers;

  FreeTrajectoryState fts = *ts.freeState();

  //if the state is outward, alongMomentum
  if (fts.position().basicVector().dot(fts.momentum().basicVector())>0) 
     navLayers = navigation.compatibleLayers(fts, alongMomentum);
  else navLayers = navigation.compatibleLayers(fts, oppositeToMomentum);

  TrajectoryStateOnSurface lastTsos;
  LogTrace(metname)<<"traj dir "<<traj.direction();

  // FIXME: protect invalid tsos?
  if (traj.lastMeasurement().updatedState().globalPosition().y() <
      traj.firstMeasurement().updatedState().globalPosition().y()) {
    lastTsos = propagatorAlong()->propagate(fts,navLayers.front()->surface());
  }
  else {
    lastTsos = propagatorOpposite()->propagate(fts,navLayers.front()->surface());
  }

  if ( !lastTsos.isValid() ) { 
      LogTrace(metname)<<"propagation failed from fts to inner cylinder";
      return;

  }
  LogTrace(metname)<<"tsos  "<<lastTsos.globalPosition();
  lastTsos.rescaleError(10.);

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {
     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(backwardUpdator()->estimator()));
     LogTrace(metname)<<"There're "<<measL.size()<<" measurements in DetLayer "
     << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId()); 

     if ( measL.empty() ) continue;

     TrajectoryMeasurement* theMeas=measFinder.findBestMeasurement(measL,propagator());

     if ( theMeas ) {

       pair<bool,TrajectoryStateOnSurface> bkresult
            = backwardUpdator()->update(theMeas, traj, propagator());
      if (bkresult.first ) {
          LogTrace(metname)<<"update ok : "<<theMeas->recHit()->globalPosition() ;

          if((*rnxtlayer)-> subDetector() == GeomDetEnumerators::DT) DTChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::CSC) CSCChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCBarrel || (*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCEndcap) RPCChamberUsedBack++;
          TotalChamberUsedBack++;
          if ( (!traj.empty()) && bkresult.second.isValid() ) 
             lastTsos = bkresult.second;
          else if (theMeas->predictedState().isValid()) 
             lastTsos = theMeas->predictedState();
        }
    }
  }
  return;
}

void CosmicMuonTrajectoryBuilder::buildSecondHalf(Trajectory& traj) {

  //C.L.:
  // the method builds trajectory in second hemisphere with pattern
  // recognition starting from an intermediate state

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  bool trajInsideOut = (traj.firstMeasurement().recHit()->globalPosition().perp()
      < traj.lastMeasurement().recHit()->globalPosition().perp()); 

  if ( trajInsideOut ) {
    LogTrace(metname)<<"inside-out: reverseTrajectory"; 
    reverseTrajectory(traj);
  }
  TrajectoryStateOnSurface tsos = traj.lastMeasurement().updatedState();
  if ( !tsos.isValid() ) tsos = traj.lastMeasurement().predictedState();
 LogTrace(metname)<<"last tsos on traj: pos: "<< tsos.globalPosition()<<" mom: "<< tsos.globalMomentum();
  TrajectoryStateOnSurface interTS = intermediateState(tsos);

  build(interTS,insideOut,traj);
  return;
}

TrajectoryStateOnSurface CosmicMuonTrajectoryBuilder::intermediateState(const TrajectoryStateOnSurface& tsos) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  PerpendicularBoundPlaneBuilder planeBuilder;
  GlobalPoint pos(0.0, 0.0, 0.0);
  BoundPlane* SteppingPlane = planeBuilder(pos,tsos.globalDirection());

  TrajectoryStateOnSurface predTsos = propagator()->propagate(tsos, *SteppingPlane);
  if ( predTsos.isValid() )
  LogTrace(metname)<<"intermediateState: a intermediate state: pos: "<<predTsos.globalPosition() << "mom: " << predTsos.globalMomentum();

  return predTsos;

}

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


bool CosmicMuonTrajectoryBuilder::selfDuplicate(const Trajectory& traj) const {

  TransientTrackingRecHit::ConstRecHitContainer hits = traj.recHits();

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
// reverse a trajectory without refitting.
// be caution that this can be only used for cosmic muons that come from above
//
void CosmicMuonTrajectoryBuilder::reverseTrajectory(Trajectory& traj) const {

  PropagationDirection newDir = (traj.firstMeasurement().recHit()->globalPosition().y()
      < traj.lastMeasurement().recHit()->globalPosition().y())
  ? oppositeToMomentum : alongMomentum;
  Trajectory newTraj(traj.seed(), newDir);
  
  //FIXME: is this method correct? or should refit?

  std::vector<TrajectoryMeasurement> meas = traj.measurements();

  for (std::vector<TrajectoryMeasurement>::reverse_iterator itm = meas.rbegin();
       itm != meas.rend(); ++itm ) {
    newTraj.push(*itm);

  }
  traj = newTraj;

}

void CosmicMuonTrajectoryBuilder::updateTrajectory(Trajectory& traj, const MuonRecHitContainer& hits) {

  // assuming traj and hits are correctly ordered

    const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";
    TrajectoryStateOnSurface lastTsos = traj.lastMeasurement().updatedState();

    if ( !lastTsos.isValid() ) return;
    LogTrace(metname)<<"LastTSOS on traj "<<lastTsos.globalPosition()<<"mom "<<lastTsos.globalMomentum();

    TrajectoryStateOnSurface predTsos = utilities()->stepPropagate(lastTsos, hits.front().get(),*propagator());
    if ( !predTsos.isValid() ) return;

    LogTrace(metname)<<"first predTSOS "<<predTsos.globalPosition()<<"mom "<<predTsos.globalMomentum();

    DetId id = hits.front()->geographicalId();
    if ( id.null() ) return;

    TrajectoryMeasurement tm = TrajectoryMeasurement(predTsos, hits.front().get(), 0, theService->detLayerGeometry()->idToLayer(id));

    pair<bool,TrajectoryStateOnSurface> result
          = backwardUpdator()->update(&tm, traj, propagator());

    if ( result.first && result.second.isValid() )  lastTsos = result.second;
    else lastTsos = predTsos;

    if ( hits.size() > 2 ) {
    for ( MuonRecHitContainer::const_iterator ihit = hits.begin() + 1;
        ihit != hits.end() - 1; ++ihit ) {

      if ( (!(**ihit).isValid()) || (!lastTsos.isValid()) ) continue;

      if ( !(**ihit).det() ) continue;

      predTsos = propagator()->propagate(lastTsos, (**ihit).det()->surface());

      if ( predTsos.isValid() ) {

        id = (*ihit)->geographicalId();
        tm = TrajectoryMeasurement(predTsos, (*ihit).get(), 0, theService->detLayerGeometry()->idToLayer(id));

        result  = backwardUpdator()->update(&tm, traj, propagator());
        if (result.first && result.second.isValid() ) lastTsos = result.second;
        else lastTsos = predTsos;
      } else LogTrace(metname)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<(*ihit)->globalPosition();
    }
   }

  if ( hits.back()->isValid() && lastTsos.isValid() ) {
      predTsos = propagator()->propagate(lastTsos, hits.back()->det()->surface());

      if ( predTsos.isValid() ) {
        id = hits.back()->geographicalId();
        tm = TrajectoryMeasurement(predTsos, hits.back().get(), 0, theService->detLayerGeometry()->idToLayer(id));

        result  = backwardUpdator()->update(&tm, traj, propagator());
        if (result.first && result.second.isValid() ) lastTsos = result.second;
        else lastTsos = predTsos;
      } else LogTrace(metname)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<hits.back()->globalPosition();

   }

}

//
// compute degree of freedom.
// used to compare quality of trajectories while estimating direction.
// the method is copied from MuonTrackLoader. 
//
double CosmicMuonTrajectoryBuilder::computeNDOF(const Trajectory& trajectory) const {

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();

  double ndof = 0.;

  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();      rechit != transRecHits.end(); ++rechit)
    if ((*rechit)->isValid()) ndof += (*rechit)->dimension();

  return std::max(ndof - 5., 0.);

}


//
// guess the direction by normalized chi2
//
void CosmicMuonTrajectoryBuilder::estimateDirection(Trajectory& traj) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

  TransientTrackingRecHit::ConstRecHitContainer hits = traj.recHits();

  TrajectoryStateOnSurface firstTSOS = traj.firstMeasurement().updatedState();

  TrajectoryStateOnSurface lastTSOS = traj.lastMeasurement().updatedState();

  if ( !firstTSOS.isValid() || !lastTSOS.isValid() ) return;

  LogTrace(metname) <<"Two ends of the traj "<<firstTSOS.globalPosition()
                    <<", "<<lastTSOS.globalPosition();

  LogTrace(metname) <<"Their mom: "<<firstTSOS.globalMomentum()
                    <<", "<<lastTSOS.globalMomentum();

  LogTrace(metname) <<"Their mom eta: "<<firstTSOS.globalMomentum().eta()
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


// print(hits);

//  vector<Trajectory> refitted = theSmoother->trajectories(traj.seed(),hits,firstTSOS);
//  std::reverse(hits.begin(), hits.end());

//  utilities()->reverseDirection(lastTSOS,&*theService->magneticField());
//  vector<Trajectory> refittedback = theSmoother->trajectories(traj.seed(),hits,lastTSOS);

//   if ( !refitted.empty() && !refittedback.empty() ) {
//     LogTrace(metname) <<"Along Mom: chi2 " << refitted.front().chiSquared()
//          << " NDOF "<<computeNDOF(refitted.front())<<endl;

//     LogTrace(metname) <<"Opposite To Mom: chi2 " << refittedback.front().chiSquared()
//          << " NDOF "<<computeNDOF(refittedback.front())<<endl;

//     float nchi2along = refitted.front().chiSquared()/computeNDOF(refitted.front());
//     float nchi2opposite = refittedback.front().chiSquared()/computeNDOF(refittedback.front());
     //flip the direction if backward is better
//     if ( nchi2along > nchi2opposite ) { 
//          LogTrace(metname) <<"flip the trajectory";
//          traj = refittedback.front();
//      }

//  }
  return;

}

