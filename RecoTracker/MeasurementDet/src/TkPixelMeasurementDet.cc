#include "RecoTracker/MeasurementDet/interface/TkPixelMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

const float TkPixelMeasurementDet::theRocWidth  = 8.1;
const float TkPixelMeasurementDet::theRocHeight = 8.1;

TkPixelMeasurementDet::TkPixelMeasurementDet( const GeomDet* gdet,
					      const PixelClusterParameterEstimator* cpe) : 
    MeasurementDet (gdet),
    theCPE(cpe),
    empty(true),
    activeThisEvent_(true), activeThisPeriod_(true)
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

  if (isActive() == false) {
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
    		InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 
		0.F));
    return result;
  }
 
  MeasurementDet::RecHitContainer allHits = recHits( stateOnThisDet);
  for (RecHitContainer::const_iterator ihit=allHits.begin();
       ihit != allHits.end(); ihit++) {
    std::pair<bool,double> diffEst = est.estimate( stateOnThisDet, **ihit);
    if ( diffEst.first) {
      result.push_back( TrajectoryMeasurement( stateOnThisDet, *ihit, 
					       diffEst.second));
    }
    //RC else delete *ihit; // we own allHits and have to delete the ones we don't return
  }
  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    TrackingRecHit::Type type = (hasBadComponents(stateOnThisDet) ? TrackingRecHit::inactive : TrackingRecHit::missing);
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
					     InvalidTransientRecHit::build(&geomDet(), type), 0.F)); 
  }
  else {
    // sort results according to estimator value
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  return result;
}

TransientTrackingRecHit::RecHitPointer
TkPixelMeasurementDet::buildRecHit( const SiPixelClusterRef & cluster,
				    const LocalTrajectoryParameters & ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( * cluster, gdu, ltp );
  return TSiPixelRecHit::build( lv.first, lv.second, &geomDet(), cluster, theCPE);
}

TkPixelMeasurementDet::RecHitContainer 
TkPixelMeasurementDet::recHits( const TrajectoryStateOnSurface& ts ) const
{
  RecHitContainer result;
  if (empty == true ) return result;
  if (isActive() == false) return result; 
  for ( const_iterator ci = detSet_.begin(); ci != detSet_.end(); ++ ci ) {
    SiPixelClusterRef cluster = edmNew::makeRefTo( handle_, ci ); 
    if (skipClusters_.find(cluster)!=skipClusters_.end())
      edm::LogWarning("TkPixelMeasurementDet")<<"skipping this cluster from last iteration";
    else
      result.push_back( buildRecHit( cluster, ts.localParameters() ) );
  }
  return result;
}

bool
TkPixelMeasurementDet::hasBadComponents( const TrajectoryStateOnSurface &tsos ) const {
    if (badRocPositions_.empty()) return false;
    LocalPoint lp = tsos.localPosition();
    LocalError le = tsos.localError().positionError();
    double dx = 3*std::sqrt(le.xx()) + theRocWidth, dy = 3*std::sqrt(le.yy()) + theRocHeight;
    for (std::vector<LocalPoint>::const_iterator it = badRocPositions_.begin(), ed = badRocPositions_.end(); it != ed; ++it) {
        if ( (std::abs(it->x() - lp.x()) < dx) &&
             (std::abs(it->y() - lp.y()) < dy) ) return true;
    } 
    return false;
}
