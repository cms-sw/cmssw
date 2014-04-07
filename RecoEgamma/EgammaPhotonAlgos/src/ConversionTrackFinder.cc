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


ConversionTrackFinder::ConversionTrackFinder(const edm::ParameterSet& conf, const BaseCkfTrajectoryBuilder *trajectoryBuilder ) :
  theCkfTrajectoryBuilder_(trajectoryBuilder),
  theInitialState_(new TransientInitialStateEstimator(conf.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters"))),
  theTrackerGeom_(0),
  theUpdator_(0),
  thePropagator_(0) 
{
  //  std::cout << " ConversionTrackFinder base CTOR " << std::endl;
  useSplitHits_ =  conf.getParameter<bool>("useHitsSplitting");

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

  theInitialState_->setEventSetup( es );
}
