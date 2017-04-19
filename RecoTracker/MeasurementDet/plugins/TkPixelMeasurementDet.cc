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
    result.add(theInactiveHit, 0.F);
    return true;
  }
  
  // do not use this as it does not account for APE...
  // auto xyLimits = est.maximalLocalDisplacement(stateOnThisDet,fastGeomDet().specificSurface());
  auto le = stateOnThisDet.localError().positionError();
  LocalError lape = static_cast<TrackerGeomDet const &>(fastGeomDet()).localAlignmentError();
  float xl = le.xx();
  float	yl = le.yy();
  if (lape.valid()) {
    xl+=lape.xx();
    yl+=lape.yy();
  }
  // 5 sigma to be on the safe side
  xl = 5.f*std::sqrt(xl);
  yl = 5.f*std::sqrt(yl);

  /*
  if (fastGeomDet().geographicalId().subdetId()<10) {
    xl = 100.f;
    yl = 100.f;
  }
  */

  auto oldSize = result.size();
  MeasurementDet::RecHitContainer && allHits = compHits(stateOnThisDet, data,xl,yl);
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
TkPixelMeasurementDet::buildRecHit( const SiPixelClusterRef & cluster,
				    const LocalTrajectoryParameters & ltp) const
{
  const GeomDetUnit& gdu(specificGeomDet());

  auto && params = cpe()->getParameters( * cluster, gdu, ltp );
  return std::make_shared<SiPixelRecHit>( std::get<0>(params), std::get<1>(params), std::get<2>(params), fastGeomDet(), cluster);
}


TkPixelMeasurementDet::RecHitContainer
TkPixelMeasurementDet::recHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data) const {
  float xl = 100.f; // larger than any detector
  float yl = 100.f;
  return compHits(ts,data,xl,yl);
}


TkPixelMeasurementDet::RecHitContainer 
TkPixelMeasurementDet::compHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data, float xl, float yl  ) const
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

  // pixel topology is rectangular, all positions are independent
  LocalVector  maxD(xl,yl,0);
  auto PMinus = specificGeomDet().specificTopology().measurementPosition(ts.localPosition()-maxD);
  auto PPlus =  specificGeomDet().specificTopology().measurementPosition(ts.localPosition()+maxD);

  int xminus = PMinus.x();
  int yminus = PMinus.y();
  int xplus = PPlus.x()+0.5f;
  int yplus = PPlus.y()+0.5f;


  // rechits are sorted in x...
  auto rightCluster = 
    std::find_if( detSet.begin(), detSet.end(), [xplus](const SiPixelCluster& cl) { return cl.minPixelRow() > xplus; });

  // std::cout << "px xlim " << xl << ' ' << xminus << '/' << xplus << ' ' << rightCluster-detSet.begin() << ',' << detSet.end()-rightCluster << std::endl;
  

  // consider only compatible clusters
 for (auto ci = detSet.begin(); ci != rightCluster; ++ci ) {    

    if (ci < begin){
      edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch.";
      return result;
    }
     unsigned int index = ci-begin;
     if (!data.pixelClustersToSkip().empty() &&  index>=data.pixelClustersToSkip().size()){
       edm::LogError("IndexMisMatch")<<"TkPixelMeasurementDet cannot create hit because of index mismatch. i.e "<<index<<" >= "<<data.pixelClustersToSkip().size();
       return result;
     }

     if (ci->maxPixelRow()<xminus) continue;
     // also check compatibility in y... (does not add much)
     if (ci->minPixelCol()>yplus) continue;
     if (ci->maxPixelCol()<yminus) continue;

     if(data.pixelClustersToSkip().empty() or (not data.pixelClustersToSkip()[index]) ) {
       SiPixelClusterRef cluster = detSet.makeRefTo( data.pixelData().handle(), ci );
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
