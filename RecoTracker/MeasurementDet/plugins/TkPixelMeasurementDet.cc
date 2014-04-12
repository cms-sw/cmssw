#include "TkPixelMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"


namespace {
  const float theRocWidth  = 8.1;
  const float theRocHeight = 8.1;
}

TkPixelMeasurementDet::TkPixelMeasurementDet( const GeomDet* gdet,
					      PxMeasurementConditionSet & conditions ) : 
    MeasurementDet (gdet),
    theDetConditions(&conditions)
  {
    if ( dynamic_cast<const PixelGeomDetUnit*>(gdet) == 0) {
      throw MeasurementDetException( "TkPixelMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
    }
  }

bool TkPixelMeasurementDet::measurements( const TrajectoryStateOnSurface& stateOnThisDet,
					  const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
					  TempMeasurements & result) const {

  if (!isActive(data)) {
    result.add(InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F);
    return true;
  }
  
  auto oldSize = result.size();
  MeasurementDet::RecHitContainer && allHits = recHits(stateOnThisDet, data);
  for (auto && hit : allHits) {
    std::pair<bool,double> diffEst = est.estimate( stateOnThisDet, *hit);
    if ( diffEst.first)
      result.add(std::move(hit), diffEst.second);
  }

  if (result.size()>oldSize) return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  bool inac = hasBadComponents(stateOnThisDet, data);
  TrackingRecHit::Type type = inac ? TrackingRecHit::inactive : TrackingRecHit::missing;
  result.add(InvalidTransientRecHit::build(&fastGeomDet(), type), 0.F);
  return inac;

}


TransientTrackingRecHit::RecHitPointer
TkPixelMeasurementDet::buildRecHit( const SiPixelClusterRef & cluster,
				    const LocalTrajectoryParameters & ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = cpe()->localParameters( * cluster, gdu, ltp );
  return TSiPixelRecHit::build( lv.first, lv.second, cpe()->rawQualityWord(), &fastGeomDet(), cluster, cpe());
}

TkPixelMeasurementDet::RecHitContainer 
TkPixelMeasurementDet::recHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data ) const
{
  RecHitContainer result;
  if (isEmpty(data.pixelData())== true ) return result;
  if (isActive(data) == false) return result;
  const SiPixelCluster* begin=0;
  if (0 != data.pixelData().handle()->data().size()) {
     begin = &(data.pixelData().handle()->data().front());
  }
  const detset & detSet = data.pixelData().detSet(index());
  result.reserve(detSet.size());
  for ( const_iterator ci = detSet.begin(); ci != detSet.end(); ++ ci ) {
    
    if (ci < begin){
      edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
     unsigned int index = ci-begin;
     if (!data.pixelClustersToSkip().empty() &&  index>=data.pixelClustersToSkip().size()){
       edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch. i.e "<<index<<" >= "<<data.pixelClustersToSkip().size();
       return result;
     }
     if(data.pixelClustersToSkip().empty() or (not data.pixelClustersToSkip()[index]) ) {
       SiPixelClusterRef cluster = edmNew::makeRefTo( data.pixelData().handle(), ci );
       result.push_back( buildRecHit( cluster, ts.localParameters() ) );
     }else{   
       LogDebug("TkPixelMeasurementDet")<<"skipping this cluster from last iteration on "<<fastGeomDet().geographicalId().rawId()<<" key: "<<index;
     }
  }
  return result;
}

bool
TkPixelMeasurementDet::hasBadComponents( const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data ) const {
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
