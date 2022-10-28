//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
//
//

#include <sstream>

ConversionTrackFinder::ConversionTrackFinder(const edm::ParameterSet& conf,
                                             const BaseCkfTrajectoryBuilder* trajectoryBuilder,
                                             edm::ConsumesCollector iC)
    : theCkfTrajectoryBuilder_(trajectoryBuilder),
      theInitialState_(new TransientInitialStateEstimator(
          conf.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters"), iC)),
      theTrackerGeom_(nullptr),
      theUpdator_(nullptr),
      thePropagator_(nullptr),
      theMeasurementTrackerToken_(iC.esConsumes(edm::ESInputTag("", theMeasurementTrackerName_))),
      theTrackerGeomToken_(iC.esConsumes()),
      thePropagatorToken_(iC.esConsumes(edm::ESInputTag("", "AnyDirectionAnalyticalPropagator")))

{
  //  std::cout << " ConversionTrackFinder base CTOR " << std::endl;
  useSplitHits_ = conf.getParameter<bool>("useHitsSplitting");
  theMeasurementTrackerName_ = conf.getParameter<std::string>("MeasurementTrackerName");
}

ConversionTrackFinder::~ConversionTrackFinder() {}

void ConversionTrackFinder::setEventSetup(const edm::EventSetup& es) {
  theMeasurementTracker_ = &es.getData(theMeasurementTrackerToken_);

  theTrackerGeom_ = &es.getData(theTrackerGeomToken_);

  thePropagator_ = es.getHandle(thePropagatorToken_);
  theInitialState_->setEventSetup(
      es, static_cast<TkTransientTrackingRecHitBuilder const*>(theCkfTrajectoryBuilder_->hitBuilder())->cloner());
}
