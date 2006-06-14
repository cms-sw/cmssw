#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu  - Purdue Univeristy
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

/* Collaborating Class Header */
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h" 
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "DataFormats/TrackReco/interface/PerigeeParameters.h"
#include "DataFormats/TrackReco/interface/PerigeeCovariance.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicNavigation.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"


CosmicMuonTrajectoryBuilder::CosmicMuonTrajectoryBuilder(const MagneticField* field) 
 : thePropagator(new SteppingHelixPropagator(field, alongMomentum)),
  theEstimator(new Chi2MeasurementEstimator(50.0, 3.0)),
  theBestMeasurementFinder(new MuonBestMeasurementFinder(thePropagator)),
//  theUpdator(new MuonTrajectoryUpdator(thePropagator)),
  theField(field)
{
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "CosmicMuonTrajectoryBuilder begin";
  theMaxChi2=150.0;
  theMaxEta=2.9;
  theSeedCollectionLabel = "MuonSeedProducer";
}

CosmicMuonTrajectoryBuilder::~CosmicMuonTrajectoryBuilder() {
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "CosmicMuonTrajectoryBuilder end";
  delete thePropagator;
  delete theEstimator;
  delete theBestMeasurementFinder; 
//  delete theUpdator;
}


std::vector<Trajectory> CosmicMuonTrajectoryBuilder::trajectories(const edm::Event& event, const edm::EventSetup& iSetup) const{

  std::vector<Trajectory> trajL;
  TrajectoryStateTransform tsTransform;
  MuonDetLayerMeasurements theLayerMeasurements("DTSegment4DProducer","CSCSegmentProducer");
  theLayerMeasurements.setEvent(event);
 
  KFUpdator theKFUpdator;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
  iSetup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry); 
    
  CosmicNavigation navigation(&*theDetLayerGeometry);
 
  edm::Handle<TrajectorySeedCollection> seedsHandle; 
  event.getByLabel(theSeedCollectionLabel,seedsHandle);
  TrajectorySeedCollection seeds = *seedsHandle;

  bool useMeas = false;  //FIXME

  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "No. of Seeds: "<<seedsHandle->size();
   int cc = 0; 
//Loop over seeds to construct trajectories
for (TrajectorySeedCollection::const_iterator seed = seeds.begin(); seed != seeds.end(); seed ++) {

  cc++;
  PTrajectoryStateOnDet ptsd1(seed->startingState());
  DetId did(ptsd1.detId());
  Plane bp = theTrackingGeometry->idToDet(did)->surface();
  TrajectoryStateOnSurface seedTsos = tsTransform.transientState(ptsd1,&bp,theField);
  TrajectoryStateOnSurface lastTsos(seedTsos);
  TrajectoryStateOnSurface secondLastTsos(seedTsos);

   vector<const DetLayer*> navLayerCBack = navigation.compatibleLayers(*(seedTsos.freeState()), oppositeToMomentum);
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "navLayerCBack size "<<navLayerCBack.size();
  if (navLayerCBack.size() == 0) break;
  Trajectory theTraj(*seed);

  int DTChamberUsedBack = 0;
  int CSCChamberUsedBack = 0;
  int RPCChamberUsedBack = 0;
  int TotalChamberUsedBack = 0;

  for ( vector<const DetLayer*>::const_iterator rnxtlayer = navLayerCBack.begin(); rnxtlayer!= navLayerCBack.end(); ++rnxtlayer) {

  if (useMeas) {
      vector<TrajectoryMeasurement> measL =
        theLayerMeasurements.measurements(*rnxtlayer, lastTsos, propagator(), *estimator());
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "measL "<<measL.size();
     if (measL.size()==0 ) continue;
     TrajectoryMeasurement* theMeas=measFinder()->findBestMeasurement(measL);

      if ( theMeas ) {
   pair<bool,TrajectoryStateOnSurface> result = updator()->update(theMeas, theTraj);
        if (result.first) {
          if((*rnxtlayer)->module()==dt) DTChamberUsedBack++;
          else if((*rnxtlayer)->module()==csc) CSCChamberUsedBack++;
          else if((*rnxtlayer)->module()==rpc) RPCChamberUsedBack++;
          TotalChamberUsedBack++;

          secondLastTsos = lastTsos;
          if ( !theTraj.empty() ) lastTsos = result.second;
          else lastTsos=theMeas->predictedState();
        }//if result.first
      }//if the meas
    }//if usemeas 

//----------------------------------------------------------------
   if (!useMeas) {//then use rechit  ! FIXME
     std::vector<MuonTransientTrackingRecHit*> rhss = theLayerMeasurements.recHits(*rnxtlayer, event);
    edm::LogInfo("CosmicMuonTrajectoryBuilder")<<"No.RH on this Det"<<rhss.size();
    if (rhss.size()==0) continue;
   MuonTransientTrackingRecHit* itt=rhss.front(); //FIXME: works since usually only one rechit

   edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "RecHit: gp: "<<itt->globalPosition()<<"dir: "<<itt->globalDirection();
  
   BoundPlane ibsur = itt->det()->surface();
   Plane isur = itt->det()->surface();
   TrajectoryStateOnSurface predTsos= thePropagator->propagate(*(lastTsos.freeState()),isur);
   if (! predTsos.isValid() ) continue;
    edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "PredictedTSOS: gp: "<<predTsos.globalPosition()<<" mom: "<<predTsos.globalMomentum();
    TrajectoryStateOnSurface updatedTsos = theKFUpdator.update(predTsos,(*itt));
    TrajectoryMeasurement* theMeas = new TrajectoryMeasurement(predTsos, &(*itt));
    if (updatedTsos.isValid()) { 
       theMeas = new TrajectoryMeasurement(predTsos,updatedTsos, &(*itt));
       edm::LogInfo ("CosmicMuonTrajectoryBuilder")<<"UpdatedTSOS: gp: "<<updatedTsos.globalPosition()<<" mom: "<<updatedTsos.globalMomentum();
    } 

  theTraj.push(*theMeas);

  if((*rnxtlayer)->module()==dt) DTChamberUsedBack++; 
  else if((*rnxtlayer)->module()==csc) CSCChamberUsedBack++; 
  else if((*rnxtlayer)->module()==rpc) RPCChamberUsedBack++; 
  TotalChamberUsedBack++;

  secondLastTsos = lastTsos;
  if ( !theTraj.empty() ) lastTsos = theMeas->updatedState(); 
  else lastTsos=theMeas->predictedState();
      } //if use rechit 
  } //for layers

  if (theTraj.isValid() && TotalChamberUsedBack>=2 && (DTChamberUsedBack+CSCChamberUsedBack)>0){

     trajL.push_back(theTraj);
     edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "found one valid trajectory! ";

    } //if traj valid

  } //for seeds
  edm::LogInfo ("CosmicMuonTrajectoryBuilder")<< "found trajectories: "<<trajL.size();
  return trajL;
}

