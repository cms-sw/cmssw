//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
//
//

#include <sstream>


ConversionTrackFinder::ConversionTrackFinder(const edm::EventSetup& es, 
					     const edm::ParameterSet& conf ) :  
  conf_(conf), 
  theCkfTrajectoryBuilder_(0), 
  theTrackerGeom_(0),
  theUpdator_(0),
  thePropagator_(0) 
{
  //  std::cout << " ConversionTrackFinder base CTOR " << std::endl;
  useSplitHits_ =  conf_.getParameter<bool>("useHitsSplitting");
  theMeasurementTrackerName_ = conf.getParameter<std::string>("MeasurementTrackerName");
}


ConversionTrackFinder::~ConversionTrackFinder() {
}


void ConversionTrackFinder::setEventSetup(const edm::EventSetup& es )   {

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(theMeasurementTrackerName_,measurementTrackerHandle);
  theMeasurementTracker_ = measurementTrackerHandle.product();

  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTrackerGeom_= trackerHandle.product();

  es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator",
					thePropagator_);
}

void ConversionTrackFinder::setTrajectoryBuilder(const edm::EventSetup& es, const BaseCkfTrajectoryBuilder & builder)   {
  theCkfTrajectoryBuilder_ = & builder;
  edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialState_.reset(new TransientInitialStateEstimator( es,  tise_params,static_cast<TkTransientTrackingRecHitBuilder const *>(builder.hitBuilder())->cloner()));
}
