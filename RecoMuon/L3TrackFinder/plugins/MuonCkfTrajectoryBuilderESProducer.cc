#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilderESProducer.h"
#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
//#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <string>
#include <memory>

using namespace edm;

MuonCkfTrajectoryBuilderESProducer::MuonCkfTrajectoryBuilderESProducer(const edm::ParameterSet & p) 
{ 
  std::string myName = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myName);
}

MuonCkfTrajectoryBuilderESProducer::~MuonCkfTrajectoryBuilderESProducer() {}

boost::shared_ptr<TrajectoryBuilder> 
MuonCkfTrajectoryBuilderESProducer::produce(const CkfComponentsRecord& iRecord)
{ 
  std::string updatorName            = pset_.getParameter<std::string>("updator");   
  std::string propagatorAlongName    = pset_.getParameter<std::string>("propagatorAlong");
  std::string propagatorOppositeName = pset_.getParameter<std::string>("propagatorOpposite");   
  std::string propagatorProximityName = pset_.getParameter<std::string>("propagatorProximity");   
  std::string estimatorName          = pset_.getParameter<std::string>("estimator"); 
  std::string recHitBuilderName      = pset_.getParameter<std::string>("TTRHBuilder");     
  std::string measurementTrackerName = pset_.getParameter<std::string>("MeasurementTrackerName");
  std::string filterName             = pset_.getParameter<std::string>("trajectoryFilterName");

  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  edm::ESHandle<Propagator>             propagatorAlongHandle;
  edm::ESHandle<Propagator>             propagatorOppositeHandle;
  edm::ESHandle<Propagator>             propagatorProximityHandle;
  edm::ESHandle<Chi2MeasurementEstimatorBase>   estimatorHandle;
  edm::ESHandle<TransientTrackingRecHitBuilder> recHitBuilderHandle;
  edm::ESHandle<MeasurementTracker>             measurementTrackerHandle;
  edm::ESHandle<TrajectoryFilter>               trajectoryFilterHandle;

  iRecord.getRecord<TrackingComponentsRecord>().get(updatorName,updatorHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(propagatorAlongName,propagatorAlongHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(propagatorOppositeName,propagatorOppositeHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(propagatorProximityName,propagatorProximityHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(estimatorName,estimatorHandle);  
  iRecord.getRecord<TransientRecHitRecord>().get(recHitBuilderName,recHitBuilderHandle);  
  iRecord.get(measurementTrackerName, measurementTrackerHandle);  
  iRecord.get(filterName,trajectoryFilterHandle);
    
  _trajectoryBuilder  = 
    boost::shared_ptr<TrajectoryBuilder>(new MuonCkfTrajectoryBuilder(pset_,
								      updatorHandle.product(),
								      propagatorAlongHandle.product(),
								      propagatorOppositeHandle.product(),
								      propagatorProximityHandle.product(),
								      estimatorHandle.product(),
								      recHitBuilderHandle.product(),
								      measurementTrackerHandle.product(),
								      trajectoryFilterHandle.product()) );  
  return _trajectoryBuilder;
}


