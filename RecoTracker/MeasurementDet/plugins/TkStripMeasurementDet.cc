#include "TkStripMeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include <typeinfo>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

TkStripMeasurementDet::TkStripMeasurementDet( const GeomDet* gdet,
					      StMeasurementDetSet & dets
					      ) : 
  MeasurementDet (gdet),theDets_(&dets), index_(-1)
  {
    if (dynamic_cast<const StripGeomDetUnit*>(gdet) == 0) {
      throw MeasurementDetException( "TkStripMeasurementDet constructed with a GeomDet which is not a StripGeomDetUnit");
    }
  }


std::vector<TrajectoryMeasurement> 
TkStripMeasurementDet::
fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		  const TrajectoryStateOnSurface&, 
		  const Propagator&, 
		  const MeasurementEstimator& est) const
{ 
  std::vector<TrajectoryMeasurement> result;

  if (!isActive()) {
    LogDebug("TkStripMeasurementDet")<<" found an inactive module "<<rawId();
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
    		InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 
		0.F));
    return result;
  }
 
  float utraj =  specificGeomDet().specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  float uerr;
  //  if (theClusterRange.first == theClusterRange.second) { // empty
  if (isEmpty()){
    LogDebug("TkStripMeasurementDet") << " DetID " << rawId() << " empty ";
    if (stateOnThisDet.hasError()){
    uerr= sqrt(specificGeomDet().specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F));
     } else { 
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::inactive), 0.F));
     }
    }else{
      result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F));
    }
    return result;
  }
  
  if(!isRegional()){//old implemetation with DetSet
 
   auto rightCluster = 
     std::find_if( detSet().begin(), detSet().end(), [utraj](const SiStripCluster& hit) { return hit.barycenter() > utraj; });
    
   if ( rightCluster != detSet().begin()) {
     // there are hits on the left of the utraj
     auto leftCluster = rightCluster;
     while ( --leftCluster >=  detSet().begin()) {
       if (isMasked(*leftCluster)) continue;
       SiStripClusterRef clusterref = edmNew::makeRefTo( handle(), leftCluster ); 
       if (accept(clusterref)){
	 RecHitContainer recHits = buildRecHits(clusterref,stateOnThisDet); 
	 bool isCompatible(false);
	 for( auto const & recHit : recHits){	  
	   std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
	   if ( diffEst.first ) {
	     result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						      diffEst.second));
	     isCompatible = true;
	   }
	 }
	 if(!isCompatible) break; // exit loop on first incompatible hit
       }
       else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<rawId()<<" key: "<<clusterref.key();
     }
   }
   for ( ; rightCluster != detSet().end(); rightCluster++) {
     if (isMasked(*rightCluster)) continue;
     SiStripClusterRef clusterref = edmNew::makeRefTo( handle(), rightCluster ); 
     if (accept(clusterref)){
       RecHitContainer recHits = buildRecHits(clusterref,stateOnThisDet); 
       bool isCompatible(false);
       for( auto const & recHit : recHits){	
	 std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
	 if ( diffEst.first ) {
	   result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						    diffEst.second));
	   isCompatible = true;
	 }
       }
       if(!isCompatible) break; // exit loop on first incompatible hit
     }
     else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on" << rawId()<<" key: "<<clusterref.key();
   }

  }// end block with DetSet
  else{

    LogDebug("TkStripMeasurementDet")<<" finding left/ right";
    result.reserve(size());
    unsigned int rightCluster = beginClusterI();
    for (; rightCluster!= endClusterI();++rightCluster){
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle(),rightCluster);
      if (clusterref->barycenter() > utraj) break;
    }
    
    unsigned int leftCluster = 1;
    for (unsigned int iReadBackWard=1; iReadBackWard<=(rightCluster-beginClusterI()) ; ++iReadBackWard){
      leftCluster=rightCluster-iReadBackWard;
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle(),leftCluster);
      if (isMasked(*clusterref)) continue;
      if (accept(clusterref)){
	RecHitContainer recHits = buildRecHits(clusterref,stateOnThisDet); 
	bool isCompatible(false);
	for(RecHitContainer::const_iterator recHit=recHits.begin();recHit!=recHits.end();++recHit){	  
	  std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, **recHit);
	  if ( diffEst.first ) {
	    result.push_back( TrajectoryMeasurement( stateOnThisDet, *recHit, 
						     diffEst.second));
	    isCompatible = true;
	  }
	}
	if(!isCompatible) break; // exit loop on first incompatible hit
      }
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterref.key();
    }
    
    
    for ( ; rightCluster != endClusterI(); ++rightCluster) {
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle(),rightCluster);
      if (isMasked(*clusterref)) continue;
      if (accept(clusterref)){
	RecHitContainer recHits = buildRecHits(clusterref,stateOnThisDet); 
	bool isCompatible(false);
	for(RecHitContainer::const_iterator recHit=recHits.begin();recHit!=recHits.end();++recHit){	  
	  std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, **recHit);
	  if ( diffEst.first ) {
	    result.push_back( TrajectoryMeasurement( stateOnThisDet, *recHit, 
						     diffEst.second));
	    isCompatible = true;
	  }
	}
	if(!isCompatible) break; // exit loop on first incompatible hit
      }
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterref.key();
    }
  }
  

  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    if (stateOnThisDet.hasError()){
    uerr= sqrt(specificGeomDet().specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, but active ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F));
     } else { 
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, and inactive ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::inactive), 0.F));
     }
    }else{
      result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F));
    }
  }
  else {
    //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " full: " << (result.size()) << " compatible hits";
    // sort results according to estimator value
    if ( result.size() > 1) {
      std::sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  return result;
}




TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{
  RecHitContainer result;
  if (isEmpty() == true) return result;
  if (isActive() == false) return result; // GIO

  if(!isRegional()){//old implemetation with DetSet
    result.reserve(detSet().size());
    for ( new_const_iterator ci = detSet().begin(); ci != detSet().end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( handle(), ci ); 
      if (accept(cluster))
	result.push_back( buildRecHit( cluster, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<rawId()<<" key: "<<cluster.key();
    }
  }else{
    result.reserve(size());
    for (unsigned int ci = beginClusterI() ; ci!= endClusterI();++ci){
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(regionalHandle(),ci);
      if (isMasked(*clusterRef)) continue;
      if (accept(clusterRef))
	result.push_back( buildRecHit( clusterRef, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterRef.key();
      }
  }
  return result;

}



void 
TkStripMeasurementDet::simpleRecHits( const TrajectoryStateOnSurface& ts, std::vector<SiStripRecHit2D> &result) const
{
  if (isEmpty() || !isActive()) return;

  if(!isRegional()){//old implemetation with DetSet
    result.reserve(detSet().size());
    for ( new_const_iterator ci = detSet().begin(); ci != detSet().end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( handle(), ci ); 
      if (accept(cluster))
	buildSimpleRecHit( cluster, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<rawId()<<" key: "<<cluster.key();
    }
  }else{
    result.reserve(size());
    for (unsigned int ci = beginClusterI() ; ci!= endClusterI();++ci){
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(regionalHandle(),ci);
      if (isMasked(*clusterRef)) continue;
      if (accept(clusterRef))
	buildSimpleRecHit( clusterRef, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterRef.key();
    }
  }
}



bool
TkStripMeasurementDet::testStrips(float utraj, float uerr) const {
    int16_t start = (int16_t) std::max<float>(utraj - 3.f*uerr, 0);
    int16_t end   = (int16_t) std::min<float>(utraj + 3.f*uerr, totalStrips());

    if (start >= end) { // which means either end <=0 or start >= totalStrips_
        /* LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<","<<
            " U = " << utraj << " +/- " << uerr << 
            "; Range [" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
            ": YOU'RE COMPLETELY OFF THE MODULE."; */
        //return false; 
        return true;  // Wolfgang thinks this way is better
                      // and solves some problems with grouped ckf
    } 

    typedef std::vector<BadStripBlock>::const_iterator BSBIT;

    int16_t bad = 0, largestBadBlock = 0;
    for (BSBIT bsbc = badStripBlocks().begin(), bsbe = badStripBlocks().end(); bsbc != bsbe; ++bsbc) {
        if (bsbc->last  < start) continue;
        if (bsbc->first > end)   break;
        int16_t thisBad = std::min(bsbc->last, end) - std::max(bsbc->first, start);
        if (thisBad > largestBadBlock) largestBadBlock = thisBad;
        bad += thisBad;
    }

    bool ok = (bad < (end-start)) && 
      (uint16_t(bad) <= badStripCuts().maxBad) && 
      (uint16_t(largestBadBlock) <= badStripCuts().maxConsecutiveBad);

//    if (bad) {   
//       edm::LogWarning("TkStripMeasurementDet") << "Testing module " << id_ <<" (subdet: "<< SiStripDetId(id_).subdetId() << ")" <<
//            " U = " << utraj << " +/- " << uerr << 
//            "; Range [" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
//            "= [" << start << "," << end << "]" <<
//            " total strips:" << (end-start) << ", good:" << (end-start-bad) << ", bad:" << bad << ", largestBadBlock: " << largestBadBlock << 
//            ". " << (ok ? "OK" : "NO"); 
//    }
    return ok;
}

