#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  class to build trajectories of muons from cosmic rays
 *  using DirectMuonNavigation
 *
 *  $Date: 2006/08/01 15:29:22 $
 *  $Revision: 1.8 $
 *  \author Chang Liu  - Purdue Univeristy
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

/* Collaborating Class Header */
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace edm;

CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const edm::ParameterSet& par) 
{
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "CosmicMuonTrajectoryBuilder begin";
  theMaxChi2 = par.getParameter<double>("MaxChi2");;
  thePropagatorName = par.getParameter<string>("Propagator");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);

  bool enableDTMeasurement = par.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  bool enableCSCMeasurement = par.getUntrackedParameter<bool>("EnableCSCMeasurement",true);
  bool enableRPCMeasurement = par.getUntrackedParameter<bool>("EnableRPCMeasurement",true);

  theLayerMeasurements= new MuonDetLayerMeasurements(enableDTMeasurement,
						     enableCSCMeasurement,
						     enableRPCMeasurement);


}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "CosmicMuonTrajectoryBuilder end";
  delete theUpdator;
  delete theBestMeasurementFinder; 
  delete theEstimator;
//  delete thePropagator;
}

void CosmicMuonTrajectoryBuilder::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  setup.get<IdealMagneticFieldRecord>().get(theField);
  setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);

  //get Propagator 
  ESHandle<Propagator> eshPropagator;
  setup.get<TrackingComponentsRecord>().get(thePropagatorName, eshPropagator);

  if(thePropagator) delete thePropagator;

  thePropagator = eshPropagator->clone();

  theBestMeasurementFinder = new MuonBestMeasurementFinder(thePropagator);
  theUpdator = new MuonTrajectoryUpdator(thePropagator, theMaxChi2, 0);

}

void CosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  theLayerMeasurements->setEvent(event);

}

MuonTrajectoryBuilder::TrajectoryContainer 
CosmicMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){

  std::vector<Trajectory*> trajL;
  TrajectoryStateTransform tsTransform;

  DirectMuonNavigation navigation(&*theDetLayerGeometry);
 
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theTrackingGeometry->idToDet(did)->surface();
  TrajectoryStateOnSurface lastTsos = tsTransform.transientState(ptsd1,&bp,&*theField);
  if ( !lastTsos.isValid() ) return trajL;

  vector<const DetLayer*> navLayerCBack = navigation.compatibleLayers(*(lastTsos.freeState()), oppositeToMomentum);
  edm::LogInfo("CosmicMuonTrajectoryBuilder")<<"found "<<navLayerCBack.size()<<" compatible DetLayers for the Seed";
  if (navLayerCBack.size() == 0) {
    return std::vector<Trajectory*>();
  }  
  Trajectory* theTraj = new Trajectory(seed);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;

  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin(); rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

     vector<TrajectoryMeasurement> measL =
        theLayerMeasurements->measurements(*rnxtlayer, lastTsos, *propagator(), *estimator());
edm::LogInfo("CosmicMuonTrajectoryBuilder")<<"measurements in DetLayer "<<measL.size();

     if (measL.size()==0 ) continue;
     TrajectoryMeasurement* theMeas=measFinder()->findBestMeasurement(measL);
     
      if ( theMeas ) {

        pair<bool,TrajectoryStateOnSurface> result
            = updator()->update(theMeas, *theTraj);

        if (result.first ) {
   edm::LogInfo("CosmicMuonTrajectoryBuilder")<< "update successfully";
          if((*rnxtlayer)-> subDetector() == GeomDetEnumerators::DT) DTChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::CSC) CSCChamberUsedBack++;
          else if((*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCBarrel || (*rnxtlayer)->subDetector() == GeomDetEnumerators::RPCEndcap) RPCChamberUsedBack++;
          TotalChamberUsedBack++;

          if ( (!theTraj->empty()) && result.second.isValid() ) 
             lastTsos = result.second;
          else if (theMeas->predictedState().isValid()) lastTsos = theMeas->predictedState();
        }
      }
  } 

  if (theTraj->isValid() && TotalChamberUsedBack >= 2 && (DTChamberUsedBack+CSCChamberUsedBack) > 0){
     trajL.push_back(theTraj);
    } 

  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "trajectories: "<<trajL.size();

  return trajL;
}

