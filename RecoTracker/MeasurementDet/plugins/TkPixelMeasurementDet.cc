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
					      const PixelClusterParameterEstimator* cpe) : 
    MeasurementDet (gdet),
    theCPE(cpe),
    skipClusters_(0),
    empty(true),
    activeThisEvent_(true), activeThisPeriod_(true)
  {
    if ( dynamic_cast<const PixelGeomDetUnit*>(gdet) == 0) {
      throw MeasurementDetException( "TkPixelMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
    }
  }

bool TkPixelMeasurementDet::measurements( const TrajectoryStateOnSurface& stateOnThisDet,
					  const MeasurementEstimator& est,
					  TempMeasurements & result) const {

  if (!isActive()) {
    result.add(InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F);
    return true;
  }
  
 
  MeasurementDet::RecHitContainer && allHits = recHits(stateOnThisDet);
  for (auto && hit : allHits) {
    std::pair<bool,double> diffEst = est.estimate( stateOnThisDet, *hit);
    if ( diffEst.first)
      result.add(std::move(hit), diffEst.second);
  }

  if ( !result.empty()) return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  bool inac = hasBadComponents(stateOnThisDet);
  TrackingRecHit::Type type = inac ? TrackingRecHit::inactive : TrackingRecHit::missing);
  result.add(InvalidTransientRecHit::build(&fastGeomDet(), type), 0.F);
  return inac;
}

}

TransientTrackingRecHit::RecHitPointer
TkPixelMeasurementDet::buildRecHit( const SiPixelClusterRef & cluster,
				    const LocalTrajectoryParameters & ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( * cluster, gdu, ltp );
  return TSiPixelRecHit::build( lv.first, lv.second, &fastGeomDet(), cluster, theCPE);
}

TkPixelMeasurementDet::RecHitContainer 
TkPixelMeasurementDet::recHits( const TrajectoryStateOnSurface& ts ) const
{
  RecHitContainer result;
  if (empty == true ) return result;
  if (isActive() == false) return result;
  const SiPixelCluster* begin=0;
  if(0!=handle_->data().size()) {
     begin = &(handle_->data().front());
  }
  result.reserve(detSet_.size());
  for ( const_iterator ci = detSet_.begin(); ci != detSet_.end(); ++ ci ) {
    
    if (ci < begin){
      edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
     unsigned int index = ci-begin;
     if (skipClusters_!=0 && skipClusters_->size()!=0 &&  index>=skipClusters_->size()){
       edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch. i.e "<<index<<" >= "<<skipClusters_->size();
       return result;
     }
     if(0==skipClusters_ or skipClusters_->empty() or (not (*skipClusters_)[index]) ) {
       SiPixelClusterRef cluster = edmNew::makeRefTo( handle_, ci );
       result.push_back( buildRecHit( cluster, ts.localParameters() ) );
     }else{   
       LogDebug("TkPixelMeasurementDet")<<"skipping this cluster from last iteration on "<<fastGeomDet().geographicalId().rawId()<<" key: "<<index;
     }
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
