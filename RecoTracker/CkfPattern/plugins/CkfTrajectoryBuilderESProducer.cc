#include "RecoTracker/CkfPattern/plugins/CkfTrajectoryBuilderESProducer.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include <string>
#include <memory>

using namespace edm;

CkfTrajectoryBuilderESProducer::CkfTrajectoryBuilderESProducer(const edm::ParameterSet & p) 
{ 
  std::string myName = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myName);
}

CkfTrajectoryBuilderESProducer::~CkfTrajectoryBuilderESProducer() {}

boost::shared_ptr<TrajectoryBuilder> 
CkfTrajectoryBuilderESProducer::produce(const CkfComponentsRecord& iRecord)
{ 
  std::string updatorName            = pset_.getParameter<std::string>("updator");   
  std::string propagatorAlongName    = pset_.getParameter<std::string>("propagatorAlong");
  std::string propagatorOppositeName = pset_.getParameter<std::string>("propagatorOpposite");   
  std::string estimatorName          = pset_.getParameter<std::string>("estimator"); 
  std::string recHitBuilderName      = pset_.getParameter<std::string>("TTRHBuilder");     
  std::string measurementTrackerName = pset_.getParameter<std::string>("MeasurementTrackerName");     
  std::string filterName = pset_.getParameter<std::string>("trajectoryFilterName");
  

  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  edm::ESHandle<Propagator>             propagatorAlongHandle;
  edm::ESHandle<Propagator>             propagatorOppositeHandle;
  edm::ESHandle<Chi2MeasurementEstimatorBase>   estimatorHandle;
  edm::ESHandle<TransientTrackingRecHitBuilder> recHitBuilderHandle;
  edm::ESHandle<MeasurementTracker>             measurementTrackerHandle;
  edm::ESHandle<TrajectoryFilter> filterHandle;

  iRecord.getRecord<TrackingComponentsRecord>().get(updatorName,updatorHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(propagatorAlongName,propagatorAlongHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(propagatorOppositeName,propagatorOppositeHandle);
  iRecord.getRecord<TrackingComponentsRecord>().get(estimatorName,estimatorHandle);  
  iRecord.getRecord<TransientRecHitRecord>().get(recHitBuilderName,recHitBuilderHandle);  
  iRecord.get(measurementTrackerName, measurementTrackerHandle);
  iRecord.get(filterName, filterHandle);

  _trajectoryBuilder  = 
    boost::shared_ptr<TrajectoryBuilder>(new CkfTrajectoryBuilder(pset_,
								  updatorHandle.product(),
								  propagatorAlongHandle.product(),
								  propagatorOppositeHandle.product(),
								  estimatorHandle.product(),
								  recHitBuilderHandle.product(),
								  measurementTrackerHandle.product(),
								  filterHandle.product()) );  
  return _trajectoryBuilder;
}


