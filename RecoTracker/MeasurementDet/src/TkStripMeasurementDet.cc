#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/MeasurementDet/interface/StripClusterAboveU.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

TkStripMeasurementDet::TkStripMeasurementDet( const GeomDet* gdet,
					      const StripClusterParameterEstimator* cpe) : 
    MeasurementDet (gdet),
    theCPE(cpe)
  {
    theStripGDU = dynamic_cast<const StripGeomDetUnit*>(gdet);
    if (theStripGDU == 0) {
      throw MeasurementDetException( "TkStripMeasurementDet constructed with a GeomDet which is not a StripGeomDetUnit");
    }
  }

std::vector<TrajectoryMeasurement> 
TkStripMeasurementDet::
fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		  const TrajectoryStateOnSurface& startingState, 
		  const Propagator&, 
		  const MeasurementEstimator& est) const
{ 
  std::vector<TrajectoryMeasurement> result;

  if (theClusterRange.first == theClusterRange.second) { // empty
    result.push_back( TrajectoryMeasurement( stateOnThisDet, new InvalidTransientRecHit(&geomDet()), 0.F));
    return result;
  }

  float utraj = 
    theStripGDU->specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();

  ClusterIterator rightCluster = 
    find_if( theClusterRange.first, theClusterRange.second, StripClusterAboveU( utraj));

  if ( rightCluster != theClusterRange.first) {
    // there are hits on the left of the utraj
    ClusterIterator leftCluster = rightCluster;
    while ( --leftCluster >= theClusterRange.first) {
      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster);
      std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
      if ( diffEst.first ) {
	result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						 diffEst.second));
      }
      else break; // exit loop on first incompatible hit
    }
  }

  for ( ; rightCluster != theClusterRange.second; rightCluster++) {
    TransientTrackingRecHit* recHit = buildRecHit( *rightCluster);
    std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
    if ( diffEst.first) {
      result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
					       diffEst.second));
    }
    else break; // exit loop on first incompatible hit
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

TransientTrackingRecHit* 
TkStripMeasurementDet::buildRecHit( const SiStripCluster& cluster) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( cluster, gdu);
  std::vector<const SiStripCluster*> clustvec(1, &cluster);
  return new TSiStripRecHit2DLocalPos( &geomDet(), 
				       new SiStripRecHit2DLocalPos( lv.first, lv.second,
								    geomDet().geographicalId(),
								    clustvec));
								    
}

TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface&) const
{
  RecHitContainer result;

  // FIXME: should get the angles from the TSOS and pass them to buildRecHit!

  for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
    result.push_back( buildRecHit( *ci));
  }
  return result;
}
