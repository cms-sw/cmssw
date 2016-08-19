#include "TkPhase2OTMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

//FIXME:just temporary solution for phase2!
/*
namespace {
  const float theRocWidth  = 8.1;
  const float theRocHeight = 8.1;
}
*/

TkPhase2OTMeasurementDet::TkPhase2OTMeasurementDet( const GeomDet* gdet,
					      Phase2OTMeasurementConditionSet & conditions ) : 
    MeasurementDet (gdet),
    theDetConditions(&conditions)
  {
    if ( dynamic_cast<const PixelGeomDetUnit*>(gdet) == 0) {
      throw MeasurementDetException( "TkPhase2OTMeasurementDet constructed with a GeomDet which is not a PixelGeomDetUnit");
    }
  }

bool TkPhase2OTMeasurementDet::measurements( const TrajectoryStateOnSurface& stateOnThisDet,
					  const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
					  TempMeasurements & result) const {


  if (!isActive(data)) {
    result.add(theInactiveHit, 0.F);
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
  result.add(inac ? theInactiveHit : theMissingHit, 0.F);
  return inac;

}

TrackingRecHit::RecHitPointer
TkPhase2OTMeasurementDet::buildRecHit( const Phase2TrackerCluster1DRef & cluster,
				    const LocalTrajectoryParameters & ltp) const
{

  const PixelGeomDetUnit& gdu( specificGeomDet() );
  auto && params = cpe()->localParameters( *cluster, gdu );

  return std::make_shared<Phase2TrackerRecHit1D>( params.first, params.second, fastGeomDet(), cluster);

}

TkPhase2OTMeasurementDet::RecHitContainer 
TkPhase2OTMeasurementDet::recHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data ) const
{
  RecHitContainer result;
  if (isEmpty(data.phase2OTData())== true ) return result;
  if (isActive(data) == false) return result;
  const Phase2TrackerCluster1D* begin=0;
  if (0 != data.phase2OTData().handle()->data().size()) {
     begin = &(data.phase2OTData().handle()->data().front());
  }
  const detset & detSet = data.phase2OTData().detSet(index());
  result.reserve(detSet.size());
  for ( const_iterator ci = detSet.begin(); ci != detSet.end(); ++ ci ) {
    
    if (ci < begin){
      edm::LogError("IndexMisMatch")<<"TkPhase2OTMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
     unsigned int index = ci-begin;
     if (!data.phase2OTClustersToSkip().empty() &&  index>=data.phase2OTClustersToSkip().size()){
       edm::LogError("IndexMisMatch")<<"TkPhase2OTMeasurementDet cannot create hit because of index mismatch. i.e "<<index<<" >= "<<data.phase2OTClustersToSkip().size();
       return result;
     }
     if(data.phase2OTClustersToSkip().empty() or (not data.phase2OTClustersToSkip()[index]) ) {
       Phase2TrackerCluster1DRef cluster = detSet.makeRefTo( data.phase2OTData().handle(), ci );
       result.push_back( buildRecHit( cluster, ts.localParameters() ) );
     }else{   
       LogDebug("TkPhase2OTMeasurementDet")<<"skipping this cluster from last iteration on "<<fastGeomDet().geographicalId().rawId()<<" key: "<<index;
     }
  }
  return result;
}

//FIXME:just temporary solution for phase2!
bool
TkPhase2OTMeasurementDet::hasBadComponents( const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data ) const {
/*
    if (badRocPositions_.empty()) return false;
    LocalPoint lp = tsos.localPosition();
    LocalError le = tsos.localError().positionError();
    double dx = 3*std::sqrt(le.xx()) + theRocWidth, dy = 3*std::sqrt(le.yy()) + theRocHeight;
    for (std::vector<LocalPoint>::const_iterator it = badRocPositions_.begin(), ed = badRocPositions_.end(); it != ed; ++it) {
        if ( (std::abs(it->x() - lp.x()) < dx) &&
             (std::abs(it->y() - lp.y()) < dy) ) return true;
    } 
*/
    return false;
}
