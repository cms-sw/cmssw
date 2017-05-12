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
  
  if (recHits(stateOnThisDet,est,data,result.hits,result.distances)) return true;

  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
  bool inac = hasBadComponents(stateOnThisDet, data);
  result.add(inac ? theInactiveHit : theMissingHit, 0.F);
  return inac;

}

bool TkPhase2OTMeasurementDet::recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
                                        RecHitContainer & result, std::vector<float> & diffs) const {

  if unlikely( (!isActive(data)) || isEmpty(data.phase2OTData()) ) return false;

  auto oldSize = result.size();

  int utraj =  specificGeomDet().specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  const detset & detSet = data.phase2OTData().detSet(index()); 
  auto begin = &(data.phase2OTData().handle()->data().front());
  auto reject = [&](auto ci)-> bool { return (!data.phase2OTClustersToSkip().empty()) && data.phase2OTClustersToSkip()[ci-begin];};

  /// in principle we can use the usual 5 sigma cut from the Traj to identify the column.... 
  // auto const nc = specificGeomDet().specificTopology().ncolumns();
  auto firstCluster = detSet.begin();
  while (firstCluster!=detSet.end()) {
    auto const col = firstCluster->column();
    auto lastCluster =
    std::find_if( firstCluster, detSet.end(), [col](const Phase2TrackerCluster1D& hit) { return hit.column() != col; });

    auto rightCluster = 
      std::find_if( firstCluster, lastCluster, [utraj](const Phase2TrackerCluster1D& hit) { return int(hit.firstStrip()) > utraj; });

    if ( rightCluster != firstCluster) {
     // there are hits on the left of the utraj
     auto leftCluster = rightCluster;
     while ( --leftCluster >=  firstCluster) {
       if(reject(leftCluster)) continue;
       Phase2TrackerCluster1DRef cluster = detSet.makeRefTo( data.phase2OTData().handle(), leftCluster);
       auto hit = buildRecHit( cluster, stateOnThisDet.localParameters() );
       auto diffEst = est.estimate( stateOnThisDet, *hit); 
       if ( !diffEst.first ) break; // exit loop on first incompatible hit
       result.push_back(hit);
       diffs.push_back(diffEst.second);
     }
    }
    for ( ; rightCluster != lastCluster; rightCluster++) {
       if(reject(rightCluster)) continue;
       Phase2TrackerCluster1DRef cluster = detSet.makeRefTo( data.phase2OTData().handle(), rightCluster);
       auto hit = buildRecHit( cluster, stateOnThisDet.localParameters() );
       auto diffEst = est.estimate( stateOnThisDet, *hit);
       if ( !diffEst.first ) break; // exit loop on first incompatible hit
       result.push_back(hit);
       diffs.push_back(diffEst.second);
     }
     firstCluster = lastCluster;
   } // loop over columns 
   return result.size()>oldSize;
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
  if (isEmpty(data.phase2OTData())) return result;
  if (!isActive(data)) return result;
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
