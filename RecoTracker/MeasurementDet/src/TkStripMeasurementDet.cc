#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/MeasurementDet/interface/StripClusterAboveU.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include <typeinfo>
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorForTrackerHits.h"

TkStripMeasurementDet::TkStripMeasurementDet( const GeomDet* gdet,
					      const StripClusterParameterEstimator* cpe,
					      bool regional) : 
    MeasurementDet (gdet),
    theCPE(cpe),
    empty(true),
    isRegional(regional)
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
		  const MeasurementEstimator& anEstimator) const
{ 
  std::vector<TrajectoryMeasurement> result;

  const MeasurementEstimator *est = & anEstimator;
 
  if (active_ == false) {
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
    		InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 
		0.F));
   // edm::LogInfo("[*GIO*] TkStrMD") << " DetID " << (theStripGDU->geographicalId())() << " inactive";
    return result;
  }
 
  float utraj =  theStripGDU->specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  float uerr;
  //  if (theClusterRange.first == theClusterRange.second) { // empty
  if (empty  == true){
    uerr= sqrt(theStripGDU->specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
     }
    return result;
  }
  
  if ( typeid(anEstimator) == typeid(Chi2MeasurementEstimator&) ) {
      est = new Chi2MeasurementEstimatorForTrackerHits(static_cast<const Chi2MeasurementEstimatorBase&>(anEstimator));
  }

  if(!isRegional){//old implemetation with DetSet
    const_iterator rightCluster = 
      find_if( detSet_->begin(), detSet_->end(), StripClusterAboveU( utraj));

    if ( rightCluster != detSet_->begin()) {
      // there are hits on the left of the utraj
      const_iterator leftCluster = rightCluster;
      while ( --leftCluster >=  detSet_->begin()) {
	//      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster, 
	SiStripClusterRef clusterref = edm::makeRefTo( handle_, leftCluster->geographicalId(), leftCluster ); 
	TransientTrackingRecHit::RecHitPointer recHit = buildRecHit(clusterref, 
								    stateOnThisDet.localParameters());
	std::pair<bool,double> diffEst = est->estimate(stateOnThisDet, *recHit);
	if ( diffEst.first ) {
	  result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						   diffEst.second));
	}
	else break; // exit loop on first incompatible hit
      }
    }
    
    for ( ; rightCluster != detSet_->end(); rightCluster++) {
      SiStripClusterRef clusterref = edm::makeRefTo( handle_, rightCluster->geographicalId(), rightCluster ); 
      TransientTrackingRecHit::RecHitPointer recHit = buildRecHit( clusterref, 
								   stateOnThisDet.localParameters());
      std::pair<bool,double> diffEst = est->estimate(stateOnThisDet, *recHit);
      if ( diffEst.first) {
	result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						 diffEst.second));
      }
      else break; // exit loop on first incompatible hit
    }
  }// end block with DetSet
  else{
    const_iterator rightCluster = 
      find_if( beginCluster, endCluster, StripClusterAboveU( utraj));

    if ( rightCluster != beginCluster) {
      // there are hits on the left of the utraj
      const_iterator leftCluster = rightCluster;
      while ( --leftCluster >=  beginCluster) {
	//      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster, 
	//std::cout << "=====making ref in fastMeas left " << std::endl;
	SiStripRegionalClusterRef clusterref = edm::makeRefToSiStripRefGetter(regionalHandle_,leftCluster);
	TransientTrackingRecHit::RecHitPointer recHit = buildRecHit(clusterref, 
								    stateOnThisDet.localParameters());
	std::pair<bool,double> diffEst = est->estimate(stateOnThisDet, *recHit);
	if ( diffEst.first ) {
	  result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						   diffEst.second));
	}
	else break; // exit loop on first incompatible hit
      }
    }
    
    for ( ; rightCluster != endCluster; rightCluster++) {
      //std::cout << "=====making ref in fastMeas rigth " << std::endl;
      SiStripRegionalClusterRef clusterref = edm::makeRefToSiStripRefGetter(regionalHandle_,rightCluster);
      TransientTrackingRecHit::RecHitPointer recHit = buildRecHit( clusterref, 
								   stateOnThisDet.localParameters());
      std::pair<bool,double> diffEst = est->estimate(stateOnThisDet, *recHit);
      if ( diffEst.first) {
	result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						 diffEst.second));
      }
      else break; // exit loop on first incompatible hit
    }
  }


  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    uerr= sqrt(theStripGDU->specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
     }
  }
  else {
    // sort results according to estimator value
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  if (est != & anEstimator) delete est;
  return result;
}

TransientTrackingRecHit::RecHitPointer
TkStripMeasurementDet::buildRecHit( const SiStripClusterRef& cluster,
				    const LocalTrajectoryParameters& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( *cluster, gdu, ltp);
  return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &geomDet(), cluster, theCPE);

								    
}

TransientTrackingRecHit::RecHitPointer
TkStripMeasurementDet::buildRecHit( const SiStripRegionalClusterRef& cluster,
				    const LocalTrajectoryParameters& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( *cluster, gdu, ltp);
  return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &geomDet(), cluster, theCPE);

								    
}


TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{
  RecHitContainer result;
  if (empty == true) return result;

  if(!isRegional){//old implemetation with DetSet
    for ( const_iterator ci = detSet_->data.begin(); ci != detSet_->data.end(); ++ ci ) {
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edm::makeRefTo( handle_, id_, ci ); 
      result.push_back( buildRecHit( cluster, ts.localParameters()));
    }
  }else{
    for (const_iterator ci = beginCluster ; ci != endCluster; ci++) {      
      SiStripRegionalClusterRef clusterRef = edm::makeRefToSiStripRefGetter(regionalHandle_,ci);     
      result.push_back( buildRecHit( clusterRef, ts.localParameters()));
    }
  }
  return result;

}

bool
TkStripMeasurementDet::testStrips(float utraj, float uerr) const {
    int istart = (int) (utraj - 3*uerr); if (istart < 0) istart = 0;
    SiStripNoises::ContainerIterator start = stripNoises_.first + istart;
    int iend = (int) (utraj + 3*uerr); 
    SiStripNoises::ContainerIterator end = stripNoises_.first + iend;
    if (end > stripNoises_.second) end = stripNoises_.second;

    int found = 0, off = 0;
    while (start < end) {
        found++;
        if (*start < 0) off++;
        start++;
    }
    // std::cout << "[*GIO*] DetID " << (theStripGDU->geographicalId())() << "  u = (" << utraj << " +/- " << uerr << "), bad/tot = " << off << "/" << found << "\n";
    return !(off > 0 && off == found); //to be tuned
}
void
TkStripMeasurementDet::setNoises(const SiStripNoises::Range noises) 
{
  stripNoises_ = noises;
/*// count bad strips
  int found = 0, off = 0;
  for (SiStripNoises::ContainerIterator it = stripNoises_.first;
        it != stripNoises_.second; it++) {
        found++; if (*it < 0) off++;
  }
  //std::cout << "[*GIO*] DetID " << mydetid << ": bad strips = " << off << "/" << found << "\n";
*/
}
