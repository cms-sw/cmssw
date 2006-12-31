#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of muons from cosmic rays
 *  using DirectMuonNavigation
 *
 *  $Date: 2006/11/30 17:56:52 $
 *  $Revision: 1.19 $
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
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"
#include "Geometry/Surface/interface/PlaneBuilder.h"

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
  
  theUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet, recoMuon::insideOut);

  ParameterSet muonBackwardUpdatorPSet = par.getParameter<ParameterSet>("BackwardMuonTrajectoryUpdatorParameters");

  theBKUpdator = new MuonTrajectoryUpdator(muonBackwardUpdatorPSet, recoMuon::outsideIn);

  theCrossingMuonFlag = par.getUntrackedParameter<bool>("BuildCrossingMuon",true);

}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {
  const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";
  LogDebug(metname)<< "CosmicMuonTrajectoryBuilder dtor called";
  if (theUpdator) delete theUpdator;
  if (theBKUpdator) delete theBKUpdator;
  if (theLayerMeasurements) delete theLayerMeasurements;
}

void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);

}

MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){

  const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";
  vector<Trajectory*> trajL;
  TrajectoryStateTransform tsTransform;
  MuonPatternRecoDumper debug;

  DirectMuonNavigation navigation((theService->detLayerGeometry()));
  MuonBestMeasurementFinder measFinder;
 
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = tsTransform.transientState(ptsd1,&bp,&*theService->magneticField());
      LogDebug(metname) << "Trajectory State on Surface of Seed";
      LogDebug(metname)<<"mom: "<<lastTsos.globalMomentum();
      LogDebug(metname)<<"pos: " <<lastTsos.globalPosition();
  
  vector<const DetLayer*> navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  LogDebug(metname)<<"found "<<navLayerCBack.size()<<" compatible DetLayers for the Seed";
  if (navLayerCBack.empty()) return trajL;
  
  vector<DetWithState> detsWithStates;
  LogDebug(metname) <<"Compatible layers:";
  for( vector<const DetLayer*>::const_iterator layer = navLayerCBack.begin();
       layer != navLayerCBack.end(); layer++){
    LogDebug(metname)<< debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId()) 
                     << debug.dumpLayer(*layer);
  }

  detsWithStates = navLayerCBack.front()->compatibleDets(lastTsos, *propagator(), *(updator()->estimator()));
  LogDebug(metname)<<"Number of compatible dets: "<<detsWithStates.size()<<endl;

  if( !detsWithStates.empty() ){
    // get the updated TSOS
    if ( detsWithStates.front().second.isValid() ) {
      LogDebug(metname)<<"New starting TSOS is on det: "<<endl;
      LogDebug(metname) << debug.dumpMuonId(detsWithStates.front().first->geographicalId())
                        << debug.dumpLayer(navLayerCBack.front());
      LogDebug(metname) << "Trajectory State on Surface after extrapolation";
      lastTsos = detsWithStates.front().second;
      LogDebug(metname)<<"mom: "<<lastTsos.globalMomentum();
      LogDebug(metname)<<"pos: " << lastTsos.globalPosition();
    }
  }

  TrajectoryStateOnSurface secondLast = lastTsos;
  if ( !lastTsos.isValid() ) return trajL;
  lastTsos.rescaleError(10.0);

  Trajectory* theTraj = new Trajectory(seed,alongMomentum);
  navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), alongMomentum);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;
  MuonTransientTrackingRecHit::MuonRecHitContainer allUnusedHits;

  LogDebug(metname)<<"Begin forward refitting";
  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin(); rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(updator()->estimator()));

     LogDebug(metname)<<"There're "<<measL.size()<<" measurements in DetLayer "
     << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId());

     if ( measL.empty() ) continue;

     TrajectoryMeasurement* theMeas=measFinder.findBestMeasurement(measL,propagator());

     if ( theMeas ) {

        pair<bool,TrajectoryStateOnSurface> result
            = updator()->update(theMeas, *theTraj, propagator());

        if (result.first ) {
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

  if (!theTraj->isValid() || TotalChamberUsedBack < 2 || (DTChamberUsedBack+CSCChamberUsedBack) == 0) 
  return trajL;

  if ( !lastTsos.isValid() ) {
     return trajL;
  }

  delete theTraj;

  //if got good trajectory, then do backward refitting
  DTChamberUsedBack = 0;
  CSCChamberUsedBack = 0;
  RPCChamberUsedBack = 0;
  TotalChamberUsedBack = 0;

  Trajectory* myTraj = new Trajectory(seed, oppositeToMomentum);

  // set fit direction for MuonTrajectoryUpdator
  // it's not the same as propagation!!!

  GlobalPoint lastPos = lastTsos.globalPosition();
  GlobalPoint secondLastPos = secondLast.globalPosition();
  GlobalVector momDir(secondLastPos.x()-lastPos.x(),
                      secondLastPos.y()-lastPos.y(),
                      secondLastPos.z()-lastPos.z());
//  LogDebug("CosmicMuonTrajectoryBuilder")<<"lastTsos"<<lastPos;
//  LogDebug("CosmicMuonTrajectoryBuilder")<<"secondLast"<<secondLastPos;
//  LogDebug("CosmicMuonTrajectoryBuilder")<<"momDir"<<momDir;
  if ( lastPos.x() * momDir.x()
      +lastPos.y() * momDir.y()
      +lastPos.z() * momDir.z() > 0 ){
//      LogDebug("CosmicMuonTrajectoryBuilder")<<"Fit direction changed to insideOut";
      theBKUpdator->setFitDirection(recoMuon::insideOut);
    } else theBKUpdator->setFitDirection(recoMuon::outsideIn);

  navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);
  LogDebug(metname)<<"Begin backward refitting";

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin();
      rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(backwardUpdator()->estimator()));
     LogDebug(metname)<<"There're "<<measL.size()<<" measurements in DetLayer "
     << debug.dumpMuonId((*rnxtlayer)->basicComponents().front()->geographicalId()); 

     if ( measL.empty() ) continue;

     TrajectoryMeasurement* theMeas=measFinder.findBestMeasurement(measL,propagator());

     if ( theMeas ) {
      // if the part change, we need to reconsider the fit direction

       if (rnxtlayer != navLayerCBack.begin()) {
         vector<const DetLayer*>::const_iterator lastlayer = rnxtlayer;
         lastlayer--;

         if((*rnxtlayer)->location() != (*lastlayer)->location() ) {

            lastPos = lastTsos.globalPosition();
            GlobalPoint thisPos = theMeas->predictedState().globalPosition();
            GlobalVector momDir(thisPos.x()-lastPos.x(),
                                thisPos.y()-lastPos.y(),
                                thisPos.z()-lastPos.z());
//          LogDebug("CosmicMuonTrajectoryBuilder")<<"momDir "<<momDir;

            if ( momDir.mag() > 0.01 ) { //if lastTsos is on the surface, no need
              if ( thisPos.x() * momDir.x() 
                  +thisPos.y() * momDir.y()
                  +thisPos.z() * momDir.z() > 0 ){
                   theBKUpdator->setFitDirection(recoMuon::insideOut);
                } else theBKUpdator->setFitDirection(recoMuon::outsideIn);
            }
          }
       }
//       if (theBKUpdator->fitDirection() == recoMuon::insideOut) 
//          LogDebug("CosmicMuonTrajectoryBuilder")<<"Fit direction insideOut";
//       else LogDebug("CosmicMuonTrajectoryBuilder")<<"Fit direction outsideIn";

       pair<bool,TrajectoryStateOnSurface> bkresult
            = backwardUpdator()->update(theMeas, *myTraj, propagator());

       if (bkresult.first ) {
          if((*rnxtlayer)-> subDetector() == GeomDetEnumerators::DT) DTChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::CSC) CSCChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCBarrel || (*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCEndcap) RPCChamberUsedBack++;
          TotalChamberUsedBack++;
          if ( theCrossingMuonFlag ) {
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
  LogDebug(metname) << "Used RecHits: "<<hits.size();
  print(hits);
  LogDebug(metname) << "== End of Used RecHits == ";
  LogDebug(metname)<<"all unused RecHits: "<<allUnusedHits.size();
  if ( theCrossingMuonFlag && allUnusedHits.size() > 2 ) {
    if ( (myTraj->lastLayer()->location() == GeomDetEnumerators::barrel ) ||
         (myTraj->firstMeasurement().layer()->location() == GeomDetEnumerators::barrel ) ) {
   //   print(allUnusedHits);
   //   LogDebug(metname)<<"== End of Unused RecHits ==";
      selectHits(allUnusedHits);
      LogDebug(metname)<<"all unused RecHits after selection: "<<allUnusedHits.size();
      print(allUnusedHits);
      LogDebug(metname)<<"== End of Unused RecHits ==";

      LogDebug(metname)<<"Exploring unused RecHits...";

      explore(*myTraj, allUnusedHits);

      hits = myTraj->recHits();
      LogDebug(metname) << "After explore: Used RecHits: "<<hits.size();
      print(hits);
      LogDebug(metname) << "== End of Used RecHits == ";
    }
  }

  if (myTraj->isValid() && TotalChamberUsedBack >= 2 && (DTChamberUsedBack+CSCChamberUsedBack) > 0){
     trajL.clear();
     trajL.push_back(myTraj);
  }

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

void CosmicMuonTrajectoryBuilder::print(const MuonTransientTrackingRecHit::MuonRecHitContainer& hits) const {

    const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";

    for (MuonRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug(metname) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug(metname)
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector();
  }

}

void CosmicMuonTrajectoryBuilder::print(const TransientTrackingRecHit::ConstRecHitContainer& hits) const {

    const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";

    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug(metname) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug(metname)
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector();
  }

}

void CosmicMuonTrajectoryBuilder::explore(Trajectory& traj, MuonRecHitContainer& hits) {

  //C.L.:
  //the method determine which side of the trajectory the unused hits located
  //choose the end of Trajectory as startingTSOS which is closer to hits
  //in previous step, we know that
  //rechits in trajectory and unused rechits should have same 
  //(inside-out/outside-in) direction
  //and to combine, we must have trajectory outside-in and hits inside-out
  //Thus
  //if inside-out, reverse traj
  //if outside-out, reverse unused rechits 

  //any special consideration for endcap??

  const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";

  bool trajInsideOut = (traj.firstMeasurement().recHit()->globalPosition().perp()
      < traj.lastMeasurement().recHit()->globalPosition().perp()); 

  bool hitsInsideOut = (hits.front()->globalPosition().perp()
      < hits.back()->globalPosition().perp()); 

  theBKUpdator->setFitDirection(recoMuon::insideOut);

  if ( trajInsideOut && hitsInsideOut ) {
    LogDebug(metname)<<"inside-out: reverseTrajectory"; 
    reverseTrajectory(traj);
    updateTrajectory(traj,hits); 

  } else if ( (!trajInsideOut) && (!hitsInsideOut)) {
    //both outside-in 
    //fit with reversed hits
    LogDebug(metname)<<"outside-in: reverse hits";
    std::reverse(hits.begin(), hits.end()); 
    updateTrajectory(traj,hits);
  
  } else {
    LogDebug(metname)<<"Error: traj and hits have different directions"; //FIXME
  } 
  return;
}


TrajectoryStateOnSurface CosmicMuonTrajectoryBuilder::stepPropagate(const TrajectoryStateOnSurface& tsos,
                                              const ConstRecHitPointer& hit) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";

  GlobalPoint start = tsos.globalPosition();
  GlobalPoint dest = hit->globalPosition();
  GlobalVector StepVector = dest - start;
  GlobalVector UnitStepVector = StepVector.unit();
  GlobalPoint GP =start;
  TrajectoryStateOnSurface result(tsos);
  float totalDis = StepVector.mag();
  LogDebug(metname)<<"stepPropagate: propagate from: "<<start<<" to "<<dest;
  LogDebug(metname)<<"stepPropagate: their distance: "<<totalDis;

  int steps = 5; // need to optimize

  float oneStep = totalDis/steps;
  Basic3DVector<float> Basic3DV(StepVector.x(),StepVector.y(),StepVector.z());
  for ( int istep = 0 ; istep < steps - 1 ; istep++) {
        GP += oneStep*UnitStepVector;
        Surface::PositionType pos(GP.x(),GP.y(),GP.z());
        LogDebug(metname)<<"stepPropagate: a middle plane: "<<pos;
        Surface::RotationType rot( Basic3DV , float(0));
        PlaneBuilder::ReturnType SteppingPlane = PlaneBuilder().plane(pos,rot);
        TrajectoryStateOnSurface predTsos = propagator()->propagate( result, *SteppingPlane);
        if (predTsos.isValid()) {
            result=predTsos;
            LogDebug(metname)<<"result "<< result.globalPosition();
          }
 }

  TrajectoryStateOnSurface predTsos = propagator()->propagate( result, hit->det()->surface());
  if (predTsos.isValid()) result=predTsos;

  return result;
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

void CosmicMuonTrajectoryBuilder::reverseTrajectory(Trajectory& traj) const {

  PropagationDirection newDir = (traj.direction() == alongMomentum) ? oppositeToMomentum : alongMomentum;
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

    const std::string metname = "Muon|RecoMuon|CosmicMuonTrajectoryBuilder";
    TrajectoryStateOnSurface lastTsos = traj.lastMeasurement().updatedState();

    if ( !lastTsos.isValid() ) return;
    LogDebug(metname)<<"LastTSOS on traj "<<lastTsos.globalPosition()<<"mom "<<lastTsos.globalMomentum();

    TrajectoryStateOnSurface predTsos = stepPropagate(lastTsos, hits.front().get());
    if ( !predTsos.isValid() ) return;

    LogDebug(metname)<<"first predTSOS "<<predTsos.globalPosition()<<"mom "<<predTsos.globalMomentum();

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
      } else LogDebug(metname)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<(*ihit)->globalPosition();
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
      } else LogDebug(metname)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<hits.back()->globalPosition();

   }

}
