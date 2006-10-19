#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of muons from cosmic rays
 *  using DirectMuonNavigation
 *
 *  $Date: 2006/09/24 18:30:43 $
 *  $Revision: 1.17 $
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

using namespace edm;
using namespace std;

CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const edm::ParameterSet& par, const MuonServiceProxy*service):theService(service) { 

  thePropagatorName = par.getParameter<string>("Propagator");

  bool enableDTMeasurement = par.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  bool enableCSCMeasurement = par.getUntrackedParameter<bool>("EnableCSCMeasurement",true);
  bool enableRPCMeasurement = par.getUntrackedParameter<bool>("EnableRPCMeasurement",true);

  theLayerMeasurements= new MuonDetLayerMeasurements(enableDTMeasurement,
						     enableCSCMeasurement,
						     enableRPCMeasurement);

  // FIXME: check if the propagator is updated each event!!!  
  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  
  theUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet, recoMuon::insideOut);

  ParameterSet muonBackwardUpdatorPSet = par.getParameter<ParameterSet>("BackwardMuonTrajectoryUpdatorParameters");

  theBKUpdator = new MuonTrajectoryUpdator(muonBackwardUpdatorPSet, recoMuon::outsideIn);

}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {
  LogDebug("CosmicMuonTrajectoryBuilder")<< "CosmicMuonTrajectoryBuilder end";
  if (theUpdator) delete theUpdator;
  if (theBKUpdator) delete theBKUpdator;
  if (theLayerMeasurements) delete theLayerMeasurements;
}

void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);

}

MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){

  const std::string metname = "CosmicMuonTrajectoryBuilder";
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
  LogDebug(metname)<<"Begin forward refitting";
  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin(); rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(updator()->estimator()));
     LogDebug("CosmicMuonTrajectoryBuilder")<<"There're "<<measL.size()<<" measurements in DetLayer "
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

  //if got good  trajectory, then do backward refitting
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

            if ( momDir.mag()>0.01 ) { //if lastTsos is on the surface, no need
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

          if ( (!myTraj->empty()) && bkresult.second.isValid() ) 
             lastTsos = bkresult.second;
          else if (theMeas->predictedState().isValid()) 
             lastTsos = theMeas->predictedState();
        }
    }
  }

  if (myTraj->isValid() && TotalChamberUsedBack >= 2 && (DTChamberUsedBack+CSCChamberUsedBack) > 0){
     trajL.clear(); 
     trajL.push_back(myTraj);
    }
    LogDebug("CosmicMuonTrajectoryBuilder") << "Used RecHits: ";
    typedef TransientTrackingRecHit::ConstRecHitContainer  ConstRecHitContainer;
    ConstRecHitContainer hits = myTraj->recHits();

    for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug("CosmicMuonTrajectoryBuilder") << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug("CosmicMuonTrajectoryBuilder")
    << "r = "<<sqrt(pos.x() * pos.x() + pos.y() * pos.y())<<"  z = "<<pos.z()
    << "  dimension = " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub detector " << (*ir)->det()->subDetector();
  }

  return trajL;
}

