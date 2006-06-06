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
    theCPE(cpe),
    empty(true)
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

  //  if (theClusterRange.first == theClusterRange.second) { // empty
  if (empty  == true){
    result.push_back( TrajectoryMeasurement( stateOnThisDet, new InvalidTransientRecHit(&geomDet()), 0.F));
    return result;
  }
  
  float utraj = 
    theStripGDU->specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  
  const_iterator rightCluster = 
    find_if( detSet_->begin(), detSet_->end(), StripClusterAboveU( utraj));

  if ( rightCluster != detSet_->end()) {
    // there are hits on the left of the utraj
    const_iterator leftCluster = rightCluster;
    while ( --leftCluster >=  detSet_->begin()) {
      //      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster, 
      SiStripClusterRef clusterref = edm::makeRefTo( handle_, leftCluster->geographicalId(), leftCluster ); 
      TransientTrackingRecHit* recHit = buildRecHit(clusterref, 
						     stateOnThisDet.localParameters());
      std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
      if ( diffEst.first ) {
	result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						 diffEst.second));
      }
      else break; // exit loop on first incompatible hit
    }
  }
  
  for ( ; rightCluster != detSet_->end(); rightCluster++) {
    SiStripClusterRef clusterref = edm::makeRefTo( handle_, rightCluster->geographicalId(), rightCluster ); 
    TransientTrackingRecHit* recHit = buildRecHit( clusterref, 
						   stateOnThisDet.localParameters());
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
TkStripMeasurementDet::buildRecHit( const SiStripClusterRef& cluster,
				    const LocalTrajectoryParameters& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( *cluster, gdu, ltp);
  return new TSiStripRecHit2DLocalPos( lv.first, lv.second, &geomDet(), cluster, theCPE);

//   return new TSiStripRecHit2DLocalPos( &geomDet(), 
// 				       new SiStripRecHit2DLocalPos( lv.first, lv.second,
// 								    geomDet().geographicalId(),
// 								    clustvec));
								    
}

TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{
  RecHitContainer result;
  if (empty == true) return result;
  for ( const_iterator ci = detSet_->data.begin(); ci != detSet_->data.end(); ++ ci ) {
    // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
    SiStripClusterRef  cluster = edm::makeRefTo( handle_, id_, ci ); 
    result.push_back( buildRecHit( cluster, ts.localParameters()));
  }
  return result;
}
