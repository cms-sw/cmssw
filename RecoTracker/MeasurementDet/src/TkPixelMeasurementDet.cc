#include "RecoTracker/MeasurementDet/interface/TkPixelMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

TkPixelMeasurementDet::TkPixelMeasurementDet( const GeomDet* gdet,
					      const PixelClusterParameterEstimator* cpe) : 
    MeasurementDet (gdet),
    theCPE(cpe)
  {
    thePixelGDU = dynamic_cast<const PixelGeomDetUnit*>(gdet);
    if (thePixelGDU == 0) {
      throw MeasurementDetException( "TkPixelMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
    }
  }

std::vector<TrajectoryMeasurement> 
TkPixelMeasurementDet::fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
					 const TrajectoryStateOnSurface& startingState, 
					 const Propagator&, 
					 const MeasurementEstimator& est) const
{
  std::vector<TrajectoryMeasurement> result;

  MeasurementDet::RecHitContainer allHits = recHits( stateOnThisDet);
  for (RecHitContainer::const_iterator ihit=allHits.begin();
       ihit != allHits.end(); ihit++) {
    std::pair<bool,double> diffEst = est.estimate( stateOnThisDet, **ihit);
    if ( diffEst.first) {
      result.push_back( TrajectoryMeasurement( stateOnThisDet, *ihit, 
					       diffEst.second));
    }
  }
  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
					     new InvalidTransientRecHit(&geomDet()), 0.F)); 
  }
  else {
    // sort results according to estimator value
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  return result;
}

TransientTrackingRecHit* TkPixelMeasurementDet::buildRecHit( const SiPixelCluster& cluster) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( cluster, gdu);
  return new TSiPixelRecHit( &geomDet(), new SiPixelRecHit( lv.first, lv.second,
							    geomDet().geographicalId(),
							    &cluster));
}

TkPixelMeasurementDet::RecHitContainer 
TkPixelMeasurementDet::recHits( const TrajectoryStateOnSurface&) const
{
  RecHitContainer result;

  // FIXME: should get the angles from the TSOS and pass them to buildRecHit!

  for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
    result.push_back( buildRecHit( *ci));
  }
  return result;
}
