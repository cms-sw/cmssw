#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of cosmic muons and beam-halo muons
 *
 *
 *  $Date: 2008/06/17 18:11:07 $
 *  $Revision: 1.36 $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
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

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"

#include <algorithm>

using namespace edm;
using namespace std;

CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const edm::ParameterSet& par, const MuonServiceProxy*service):theService(service) { 

  thePropagatorName = par.getParameter<string>("Propagator");

  bool enableDTMeasurement = par.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  bool enableCSCMeasurement = par.getUntrackedParameter<bool>("EnableCSCMeasurement",true);
  bool enableRPCMeasurement = par.getUntrackedParameter<bool>("EnableRPCMeasurement",true);

//  if(enableDTMeasurement)
  InputTag DTRecSegmentLabel = par.getParameter<InputTag>("DTRecSegmentLabel");

//  if(enableCSCMeasurement)
  InputTag CSCRecSegmentLabel = par.getParameter<InputTag>("CSCRecSegmentLabel");

//  if(enableRPCMeasurement)
  InputTag RPCRecSegmentLabel = par.getParameter<InputTag>("RPCRecSegmentLabel");


  theLayerMeasurements= new MuonDetLayerMeasurements(DTRecSegmentLabel,
                                                     CSCRecSegmentLabel,
                                                     RPCRecSegmentLabel,
						     enableDTMeasurement,
						     enableCSCMeasurement,
						     enableRPCMeasurement);

  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  
  theNavigation = 0; // new DirectMuonNavigation(theService->detLayerGeometry());
  theUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet, insideOut);

  ParameterSet muonBackwardUpdatorPSet = par.getParameter<ParameterSet>("BackwardMuonTrajectoryUpdatorParameters");

  theBKUpdator = new MuonTrajectoryUpdator(muonBackwardUpdatorPSet, outsideIn);

  theTraversingMuonFlag = par.getUntrackedParameter<bool>("BuildTraversingMuon",true);

  ParameterSet smootherPSet = par.getParameter<ParameterSet>("MuonSmootherParameters");

  ParameterSet emptyPS;
  theNavigationPSet = par.getUntrackedParameter<ParameterSet>("MuonNavigationParameters", emptyPS);

  theSmoother = new CosmicMuonSmoother(smootherPSet,theService);

  theNTraversing = 0;
  theNSuccess = 0;
  category_ = "Muon|RecoMuon|CosmicMuon|CosmicMuonTrajectoryBuilder";

}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {

  LogTrace(category_)<< "CosmicMuonTrajectoryBuilder dtor called";
  if (theUpdator) delete theUpdator;
  if (theBKUpdator) delete theBKUpdator;
  if (theLayerMeasurements) delete theLayerMeasurements;
  if (theSmoother) delete theSmoother;
  if (theNavigation) delete theNavigation; 

  LogTrace(category_)<< "CosmicMuonTrajectoryBuilder Traversing: "<<theNSuccess<<"/"<<theNTraversing;

}

void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);
  if (theNavigation) delete theNavigation;
  theNavigation = new DirectMuonNavigation(theService->detLayerGeometry(), theNavigationPSet);

//  event.getByLabel("csc2DRecHits", cschits_);
//  event.getByLabel("dt1DRecHits", dthits_);
}

MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){

  vector<Trajectory*> trajL;
  TrajectoryStateTransform tsTransform;
  MuonPatternRecoDumper debug;

  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = tsTransform.transientState(ptsd1,&bp,&*theService->magneticField());
      LogTrace(category_) << "Trajectory State on Surface of Seed";
      LogTrace(category_)<<"mom: "<<lastTsos.globalMomentum();
      LogTrace(category_)<<"pos: " <<lastTsos.globalPosition();
      LogTrace(category_)<<"eta: "<<lastTsos.globalMomentum().eta();
  
  bool beamhaloFlag =  (fabs(lastTsos.globalMomentum().eta()) > 4.5);

  vector<const DetLayer*> navLayers = ( beamhaloFlag )? navigation()->compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum) : navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);

  LogTrace(category_)<<"found "<<navLayers.size()<<" compatible DetLayers for the Seed";
  if (navLayers.empty()) return trajL;
  
  vector<DetWithState> detsWithStates;
  LogTrace(category_) <<"Compatible layers:";
  for( vector<const DetLayer*>::const_iterator layer = navLayers.begin();
       layer != navLayers.end(); layer++){
    LogTrace(category_)<< debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId()) 
                     << debug.dumpLayer(*layer);
  }

  detsWithStates = navLayers.front()->compatibleDets(lastTsos, *propagator(), *(updator()->estimator()));
  LogTrace(category_)<<"Number of compatible dets: "<<detsWithStates.size()<<endl;

  if( !detsWithStates.empty() ){
    // get the updated TSOS
    if ( detsWithStates.front().second.isValid() ) {
      LogTrace(category_)<<"New starting TSOS is on det: "<<endl;
      LogTrace(category_) << debug.dumpMuonId(detsWithStates.front().first->geographicalId())
                        << debug.dumpLayer(navLayers.front());
      LogTrace(category_) << "Trajectory State on Surface after extrapolation";
      lastTsos = detsWithStates.front().second;
      LogTrace(category_)<<"mom: "<<lastTsos.globalMomentum();
      LogTrace(category_)<<"pos: " << lastTsos.globalPosition();
    }
  }

  if ( !lastTsos.isValid() ) return trajL;

  TrajectoryStateOnSurface secondLast = lastTsos;

  lastTsos.rescaleError(10.0);

  Trajectory* theTraj = new Trajectory(seed,alongMomentum);

  navLayers =  ( beamhaloFlag )  ? navigation()->compatibleEndcapLayers(*(lastTsos.freeState()), alongMomentum) : navigation()->compatibleLayers(*(lastTsos.freeState()), alongMomentum);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;
  MuonTransientTrackingRecHit::MuonRecHitContainer allUnusedHits;

  LogTrace(category_)<<"Begin forward fit "<<navLayers.size();
  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin(); rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        findBestMeasurements(*rnxtlayer, lastTsos, (updator()->estimator()));

     if ( measL.empty() ) continue;

     for (vector<TrajectoryMeasurement>::const_iterator theMeas = measL.begin(); theMeas != measL.end(); ++theMeas) {
            pair<bool,TrajectoryStateOnSurface> result
              = updator()->update((&*theMeas), *theTraj, propagator());

            if (result.first ) {
              LogTrace(category_)<<"update ok ";
              incrementChamberCounters((*rnxtlayer), DTChamberUsedBack, CSCChamberUsedBack, RPCChamberUsedBack, TotalChamberUsedBack);
              secondLast = lastTsos;
              if ( (!theTraj->empty()) && result.second.isValid() ) 
                lastTsos = result.second;
              else if ((theMeas)->predictedState().isValid()) lastTsos = (theMeas)->predictedState();
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
  } else navLayers = navigation()->compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);

  LogTrace(category_)<<"Begin backward refitting";

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        findBestMeasurements(*rnxtlayer, lastTsos, (backwardUpdator()->estimator()));

     if ( measL.empty() ) continue;

     for (vector<TrajectoryMeasurement>::const_iterator theMeas = measL.begin(); theMeas != measL.end(); ++theMeas) {
      // if the part change, we need to reconsider the fit direction
         if (rnxtlayer != navLayers.begin()) {
           vector<const DetLayer*>::const_iterator lastlayer = rnxtlayer;
           lastlayer--;

           if((*rnxtlayer)->location() != (*lastlayer)->location() ) {

              lastPos = lastTsos.globalPosition();
              GlobalPoint thisPos = (theMeas)->predictedState().globalPosition();
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
                (lastTsos.globalPosition().z() * (theMeas)->predictedState().globalPosition().z() < 0)  ) {
                    theBKUpdator->setFitDirection(insideOut);
            }

       }
//       if (theBKUpdator->fitDirection() == insideOut) 
//          LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction insideOut";
//       else LogTrace("CosmicMuonTrajectoryBuilder")<<"Fit direction outsideIn";

         pair<bool,TrajectoryStateOnSurface> bkresult
              = backwardUpdator()->update((&*theMeas), *myTraj, propagator());

         if (bkresult.first ) {

              incrementChamberCounters((*rnxtlayer), DTChamberUsedBack, CSCChamberUsedBack, RPCChamberUsedBack, TotalChamberUsedBack);

            if ( theTraversingMuonFlag ) {
              MuonRecHitContainer tmpUnusedHits = unusedHits(*rnxtlayer,*theMeas);
              allUnusedHits.insert(allUnusedHits.end(),tmpUnusedHits.begin(),tmpUnusedHits.end());
            }
            if ( (!myTraj->empty()) && bkresult.second.isValid() ) 
               lastTsos = bkresult.second;
            else if ((theMeas)->predictedState().isValid()) 
               lastTsos = (theMeas)->predictedState();
          }
       }
  }

  TransientTrackingRecHit::ConstRecHitContainer hits = myTraj->recHits();
  unsigned int nhits = hits.size(); //for debug, remove me later...

//  LogTrace(category_) << "Used RecHits before building second half: "<<hits.size();
//  print(hits);
//  LogTrace(category_) << "== End of Used RecHits == ";

  LogTrace(category_)<<"all unused RecHits: "<<allUnusedHits.size();

  if ( theTraversingMuonFlag && ( allUnusedHits.size() >= 2 ) && 
     ( ( myTraj->lastLayer()->location() == GeomDetEnumerators::barrel ) ||
       ( myTraj->firstMeasurement().layer()->location() == GeomDetEnumerators::barrel ) ) ) {
      theNTraversing++;

      LogTrace(category_)<<utilities()->print(allUnusedHits);
   //   LogTrace(category_)<<"== End of Unused RecHits ==";
//      selectHits(allUnusedHits);
//      LogTrace(category_)<<"all unused RecHits after selection: "<<allUnusedHits.size();
//      print(allUnusedHits);
//      LogTrace(category_)<<"== End of Unused RecHits ==";

      LogTrace(category_)<<"Building trajectory in second hemisphere...";

      buildSecondHalf(*myTraj);

      hits = myTraj->recHits();
      LogTrace(category_) << "After explore: Used RecHits: "<<hits.size();
      LogTrace(category_)<<utilities()->print(hits);
      LogTrace(category_) << "== End of Used RecHits == ";

      if ( hits.size() > nhits + 2 ) theNSuccess++;
      else LogTrace(category_) << "building on second hemisphere failed. ";
  }

  if (myTraj->isValid() && (!selfDuplicate(*myTraj)) && TotalChamberUsedBack >= 2 && (DTChamberUsedBack+CSCChamberUsedBack) > 0){
     LogTrace(category_) <<" traj ok ";
//     getDirectionByTime(*myTraj);

     if (beamhaloFlag) estimateDirection(*myTraj);

     for ( vector<Trajectory*>::iterator t = trajL.begin(); t != trajL.end(); ++t ) delete *t;
     trajL.clear();

//     vector<Trajectory> smoothed = theSmoother->trajectories(*myTraj);
//     if ( !smoothed.empty() )  {
//       LogTrace(category_) <<" Smoothed successfully.";

//       delete myTraj;
//       Trajectory* smthed = new Trajectory(smoothed.front());
//       trajL.push_back(smthed);
//     }
//     else {
//       LogTrace(category_) <<" Smoothing failed.";
       LogTrace(category_) <<"first "<< myTraj->firstMeasurement().updatedState()
                        <<"\n last "<<myTraj->lastMeasurement().updatedState();
       if ( myTraj->direction() == alongMomentum ) LogTrace(category_)<<"along";
       else if (myTraj->direction() == oppositeToMomentum ) LogTrace(category_)<<"opposite";
       else LogTrace(category_)<<"anyDirection";

       if ( ( myTraj->direction() == alongMomentum && 
              (myTraj->firstMeasurement().updatedState().globalPosition().y() 
              < myTraj->lastMeasurement().updatedState().globalPosition().y()))
           || (myTraj->direction() == oppositeToMomentum && 
              (myTraj->firstMeasurement().updatedState().globalPosition().y() 
              > myTraj->lastMeasurement().updatedState().globalPosition().y())) ) {
           LogTrace(category_)<<"reverse trajectory direction";
           reverseTrajectoryDirection(*myTraj); 
       }
       trajL.push_back(myTraj);
//     }
  }
  LogTrace(category_) <<" trajL ok "<<trajL.size();
//  getDirectionByTime(*myTraj);

  navLayers.clear();
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
// continue to build a trajectory starting from given trajectory state
//
void CosmicMuonTrajectoryBuilder::build(const TrajectoryStateOnSurface& ts, 
                                        const NavigationDirection& startingDir,
                                        Trajectory& traj) {

  if ( !ts.isValid() ) return;

  theBKUpdator->setFitDirection(startingDir);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;
  FreeTrajectoryState* fts = ts.freeState();
  if ( !fts ) return;

  vector<const DetLayer*> navLayers = (fts->position().basicVector().dot(fts->momentum().basicVector())>0) ? navigation()->compatibleLayers((*fts), alongMomentum) : navigation()->compatibleLayers((*fts), oppositeToMomentum);
  if (navLayers.empty()) return;

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

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayers.begin();
      rnxtlayer!= navLayers.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        findBestMeasurements(*rnxtlayer, lastTsos, (backwardUpdator()->estimator()));

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
  return;
}

void CosmicMuonTrajectoryBuilder::buildSecondHalf(Trajectory& traj) {

  //C.L.:
  // the method builds trajectory in second hemisphere with pattern
  // recognition starting from an intermediate state

  bool trajInsideOut = (traj.firstMeasurement().recHit()->globalPosition().perp()
      < traj.lastMeasurement().recHit()->globalPosition().perp()); 

  if ( trajInsideOut ) {
    LogTrace(category_)<<"inside-out: reverseTrajectory"; 
    reverseTrajectory(traj);
  }
  TrajectoryStateOnSurface tsos = traj.lastMeasurement().updatedState();
  if ( !tsos.isValid() ) tsos = traj.lastMeasurement().predictedState();
 LogTrace(category_)<<"last tsos on traj: pos: "<< tsos.globalPosition()<<" mom: "<< tsos.globalMomentum();
  TrajectoryStateOnSurface interTS = intermediateState(tsos);

  build(interTS,insideOut,traj);
  return;
}

TrajectoryStateOnSurface CosmicMuonTrajectoryBuilder::intermediateState(const TrajectoryStateOnSurface& tsos) const {


  PerpendicularBoundPlaneBuilder planeBuilder;
  GlobalPoint pos(0.0, 0.0, 0.0);
  BoundPlane* SteppingPlane = planeBuilder(pos,tsos.globalDirection());

  TrajectoryStateOnSurface predTsos = propagator()->propagate(tsos, *SteppingPlane);
  if ( predTsos.isValid() )
  LogTrace(category_)<<"intermediateState: a intermediate state: pos: "<<predTsos.globalPosition() << "mom: " << predTsos.globalMomentum();

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

void CosmicMuonTrajectoryBuilder::reverseTrajectoryDirection(Trajectory& traj) const {
   if ( traj.direction() == anyDirection ) return;
   PropagationDirection newDir = (traj.direction() == alongMomentum)? oppositeToMomentum : alongMomentum;
   Trajectory newTraj(traj.seed(), newDir);
   std::vector<TrajectoryMeasurement> meas = traj.measurements();

   for (std::vector<TrajectoryMeasurement>::const_iterator itm = meas.begin();
         itm != meas.end(); ++itm) {
      newTraj.push(*itm);
   }

   traj = newTraj;
}

void CosmicMuonTrajectoryBuilder::updateTrajectory(Trajectory& traj, const MuonRecHitContainer& hits) {

  // assuming traj and hits are correctly ordered

    TrajectoryStateOnSurface lastTsos = traj.lastMeasurement().updatedState();

    if ( !lastTsos.isValid() ) return;
    LogTrace(category_)<<"LastTSOS on traj "<<lastTsos.globalPosition()<<"mom "<<lastTsos.globalMomentum();

    TrajectoryStateOnSurface predTsos = utilities()->stepPropagate(lastTsos, hits.front().get(),*propagator());
    if ( !predTsos.isValid() ) return;

    LogTrace(category_)<<"first predTSOS "<<predTsos.globalPosition()<<"mom "<<predTsos.globalMomentum();

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
      } else LogTrace(category_)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<(*ihit)->globalPosition();
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
      } else LogTrace(category_)<<"predTsos is not valid from TSOS" <<lastTsos.globalPosition()<< " to hit "<<hits.back()->globalPosition();

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
    << "pos"<<pos
    << "radius "<<pos.perp()
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

std::vector<TrajectoryMeasurement>
CosmicMuonTrajectoryBuilder::findBestMeasurements(const DetLayer* layer,
                                             const TrajectoryStateOnSurface& tsos, const MeasurementEstimator* estimator){

  std::vector<TrajectoryMeasurement> result;
  std::vector<TrajectoryMeasurement> measurements;

  MuonBestMeasurementFinder measFinder;

  if( layer->hasGroups() ){
    std::vector<TrajectoryMeasurementGroup> measurementGroups =
      theLayerMeasurements->groupedMeasurements(layer, tsos, *propagator(), *estimator);

    for(std::vector<TrajectoryMeasurementGroup>::const_iterator tmGroupItr = measurementGroups.begin();
        tmGroupItr != measurementGroups.end(); ++tmGroupItr){
    
      measurements = tmGroupItr->measurements();
      LogTrace(category_) << "Number of Trajectory Measurement grouped layer: " << measurements.size();
      
      const TrajectoryMeasurement* bestMeasurement 
        = measFinder.findBestMeasurement(measurements,  propagator());
      
      if(bestMeasurement) result.push_back(*bestMeasurement);
    }
  } 
  else{
    measurements = theLayerMeasurements->measurements(layer, tsos, *propagator(), *estimator);
    LogTrace(category_) << "Number of Trajectory Measurement single layer: " << measurements.size();
    const TrajectoryMeasurement* bestMeasurement 
      = measFinder.findBestMeasurement(measurements, propagator());

    if(bestMeasurement) result.push_back(*bestMeasurement);
  }
  return result;
}

void CosmicMuonTrajectoryBuilder::incrementChamberCounters(const DetLayer *layer, int& dtChambers, int& cscChambers, int& rpcChambers, int& totalChambers){

  if(layer->subDetector()==GeomDetEnumerators::DT) dtChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::CSC) cscChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcChambers++; 
  totalChambers++;
}
