#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of muons from cosmic rays
 *  using DirectMuonNavigation
 *
 *  $Date: 2006/09/04 01:15:16 $
 *  $Revision: 1.14 $
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
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoMuon/TrackingTools/interface/FitDirection.h"

using namespace edm;

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

  std::vector<Trajectory*> trajL;
  TrajectoryStateTransform tsTransform;

  DirectMuonNavigation navigation((theService->detLayerGeometry()));
  MuonBestMeasurementFinder measFinder;
 
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = tsTransform.transientState(ptsd1,&bp,&*theService->magneticField());
  TrajectoryStateOnSurface secondLast = lastTsos;
  if ( !lastTsos.isValid() ) return trajL;
  lastTsos.rescaleError(10.0);
  
  vector<const DetLayer*> navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), alongMomentum);
  LogDebug("CosmicMuonTrajectoryBuilder")<<"found "<<navLayerCBack.size()<<" compatible DetLayers for the Seed";
  if (navLayerCBack.size() == 0) {
    return std::vector<Trajectory*>();
  }  
  Trajectory* theTraj = new Trajectory(seed,alongMomentum);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;

  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin(); rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(updator()->estimator()));
        LogDebug("CosmicMuonTrajectoryBuilder")<<"measurements in DetLayer "<<measL.size();

     if (measL.size()==0 ) continue;

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
  LogDebug("CosmicMuonTrajectoryBuilder")<<"lastTsos"<<lastTsos;
  LogDebug("CosmicMuonTrajectoryBuilder")<<"secondLast"<<secondLast;
  LogDebug("CosmicMuonTrajectoryBuilder")<<"momDir"<<momDir;
  if ( lastPos.x() * momDir.x()
      +lastPos.y() * momDir.y()
      +lastPos.z() * momDir.z() > 0 ){
      LogDebug("CosmicMuonTrajectoryBuilder")<<"Fit direction changed to insideOut";
      theBKUpdator->setFitDirection(recoMuon::insideOut);
    } else theBKUpdator->setFitDirection(recoMuon::outsideIn);

  navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);

  reverse(navLayerCBack.begin(),navLayerCBack.end());

  for (vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin();
      rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *(backwardUpdator()->estimator()));
     LogDebug("CosmicMuonTrajectoryBuilder")<<"measurements in DetLayer "<<measL.size();

     if (measL.size()==0 ) continue;

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

            if ( thisPos.x() * momDir.x() 
                +thisPos.y() * momDir.y()
                +thisPos.z() * momDir.z() > 0 ){
                 theBKUpdator->setFitDirection(recoMuon::insideOut);
              } else theBKUpdator->setFitDirection(recoMuon::outsideIn);
          }
       }
 
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

