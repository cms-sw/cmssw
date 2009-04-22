#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilterESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::ClusterShapeTrajectoryFilterESProducer
  (const edm::ParameterSet& iConfig)
{
  componentName = iConfig.getParameter<std::string>("ComponentName");
  
  setWhatProduced(this, componentName);
}


/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::~ClusterShapeTrajectoryFilterESProducer
  ()
{
}

/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::ReturnType
ClusterShapeTrajectoryFilterESProducer::produce
(const TrajectoryFilter::Record &iRecord)
{
  using namespace edm::es;

  edm::ESHandle<ClusterShapeHitFilter> shape;
  iRecord.get("ClusterShapeHitFilter",shape);

  // Produce the filter using the plugin factory
  ClusterShapeTrajectoryFilterESProducer::ReturnType
    aFilter(new ClusterShapeTrajectoryFilter(  shape.product()));

  return aFilter;
}
