//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
//
//

#include <sstream>


ConversionTrackFinder::ConversionTrackFinder(const edm::EventSetup& es, 
					     const edm::ParameterSet& conf ) :  
  conf_(conf), 
  theCkfTrajectoryBuilder_(0), 
  theInitialState_(0),
  theTrackerGeom_(0),
  theUpdator_(0),
  thePropagator_() 
{
  //  std::cout << " ConversionTrackFinder base CTOR " << std::endl;

  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialState_       = new TransientInitialStateEstimator( es,  tise_params);
  useSplitHits_ =  conf_.getParameter<bool>("useHitsSplitting");

  theMeasurementTrackerName_ = conf.getParameter<std::string>("MeasurementTrackerName");

}


ConversionTrackFinder::~ConversionTrackFinder() {


  delete theInitialState_;

}


void ConversionTrackFinder::setEventSetup(const edm::EventSetup& es )   {

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(theMeasurementTrackerName_,measurementTrackerHandle);
  theMeasurementTracker_ = measurementTrackerHandle.product();

  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTrackerGeom_= trackerHandle.product();

  if(thePropagatorWatcher_.check(es)) {
    edm::ESHandle<Propagator> propHandle;
    es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator",
					   propHandle);
    thePropagator_.reset(propHandle->clone());
  }

  theInitialState_->setEventSetup( es );
}

void ConversionTrackFinder::setTrajectoryBuilder(const TrajectoryBuilder & builder)   {
  theCkfTrajectoryBuilder_ = & builder;
}
