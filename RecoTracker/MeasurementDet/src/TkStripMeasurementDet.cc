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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

TkStripMeasurementDet::TkStripMeasurementDet( const GeomDet* gdet,
					      const StripClusterParameterEstimator* cpe,
					      bool regional) : 
    MeasurementDet (gdet),
    isRegional(regional),
    empty(true),
    activeThisEvent_(true), activeThisPeriod_(true),
    theCPE(cpe)
  {
    theStripGDU = dynamic_cast<const StripGeomDetUnit*>(gdet);
    if (theStripGDU == 0) {
      throw MeasurementDetException( "TkStripMeasurementDet constructed with a GeomDet which is not a StripGeomDetUnit");
    }

    //intialize the detId !
    id_ = gdet->geographicalId().rawId();
    //initalize the total number of strips
    totalStrips_ =  specificGeomDet().specificTopology().nstrips();
  }

std::vector<TrajectoryMeasurement> 
TkStripMeasurementDet::
fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		  const TrajectoryStateOnSurface& startingState, 
		  const Propagator&, 
		  const MeasurementEstimator& est) const
{ 
  std::vector<TrajectoryMeasurement> result;

  if (isActive() == false) {
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
    		InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 
		0.F));
    //    LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " inactive";
    return result;
  }
 
  float utraj =  theStripGDU->specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  float uerr;
  //  if (theClusterRange.first == theClusterRange.second) { // empty
  if (empty  == true){
    //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty ";
    if (stateOnThisDet.hasError()){
    uerr= sqrt(theStripGDU->specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
     }
    }else{
      result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
    }
    return result;
  }
  
  if(!isRegional){//old implemetation with DetSet
    new_const_iterator rightCluster = 
      std::find_if( detSet_.begin(), detSet_.end(), StripClusterAboveU( utraj)); //FIXME

    if ( rightCluster != detSet_.begin()) {
      // there are hits on the left of the utraj
      new_const_iterator leftCluster = rightCluster;
      while ( --leftCluster >=  detSet_.begin()) {
        if (isMasked(*leftCluster)) continue;
	SiStripClusterRef clusterref = edmNew::makeRefTo( handle_, leftCluster ); 
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
	else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterref.key();
      }
    }
    
    for ( ; rightCluster != detSet_.end(); rightCluster++) {
      if (isMasked(*rightCluster)) continue;
      SiStripClusterRef clusterref = edmNew::makeRefTo( handle_, rightCluster ); 
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
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterref.key();
    }
  }// end block with DetSet
  else{
    const_iterator rightCluster = 
      std::find_if( beginCluster, endCluster, StripClusterAboveU( utraj));

    if ( rightCluster != beginCluster) {
      // there are hits on the left of the utraj
      const_iterator leftCluster = rightCluster;
      while ( --leftCluster >=  beginCluster) {
        if (isMasked(*leftCluster)) continue;

	SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle_,leftCluster-regionalHandle_->begin_record());
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
	else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterref.key();
      }
    }
    
    for ( ; rightCluster != endCluster; rightCluster++) {
      if (isMasked(*rightCluster)) continue;
      //std::cout << "=====making ref in fastMeas rigth " << std::endl;
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle_,rightCluster-regionalHandle_->begin_record());
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
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterref.key();
    }
  }


  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    if (stateOnThisDet.hasError()){
    uerr= sqrt(theStripGDU->specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, but active ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, and inactive ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
     }
    }else{
      result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
    }
  }
  else {
    //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " full: " << (result.size()) << " compatible hits";
    // sort results according to estimator value
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  return result;
}


TransientTrackingRecHit::RecHitPointer
TkStripMeasurementDet::buildRecHit( const SiStripClusterRef& cluster,
				    const TrajectoryStateOnSurface& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( *cluster, gdu, ltp);
  return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &geomDet(), cluster, theCPE);
}

TransientTrackingRecHit::RecHitPointer
TkStripMeasurementDet::buildRecHit( const SiStripRegionalClusterRef& cluster,
				    const TrajectoryStateOnSurface& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  LocalValues lv = theCPE->localParameters( *cluster, gdu, ltp);
  return TSiStripRecHit2DLocalPos::build( lv.first, lv.second, &geomDet(), cluster, theCPE);
}



TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::buildRecHits( const SiStripClusterRef& cluster,
				     const TrajectoryStateOnSurface& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  VLocalValues vlv = theCPE->localParametersV( *cluster, gdu, ltp);
  RecHitContainer res;
  for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it){
    res.push_back(TSiStripRecHit2DLocalPos::build( it->first, it->second, &geomDet(), cluster, theCPE));
  }
  return res; 
}


TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::buildRecHits( const SiStripRegionalClusterRef& cluster,
				     const TrajectoryStateOnSurface& ltp) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  VLocalValues vlv = theCPE->localParametersV( *cluster, gdu, ltp);
  RecHitContainer res;
  for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it){
    res.push_back(TSiStripRecHit2DLocalPos::build( it->first, it->second, &geomDet(), cluster, theCPE));
  }
  return res; 
}



TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& ts) const
{
  RecHitContainer result;
  if (empty == true) return result;
  if (isActive() == false) return result; // GIO

  if(!isRegional){//old implemetation with DetSet
    result.reserve(detSet_.size());
    for ( new_const_iterator ci = detSet_.begin(); ci != detSet_.end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( handle_, ci ); 
      if (accept(cluster))
	result.push_back( buildRecHit( cluster, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<cluster.key();
    }
  }else{
    result.reserve(endCluster - beginCluster);
    for (const_iterator ci = beginCluster ; ci != endCluster; ci++) {      
      if (isMasked(*ci)) continue;
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(regionalHandle_,ci-regionalHandle_->begin_record());     
      if (accept(clusterRef))
	result.push_back( buildRecHit( clusterRef, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterRef.key();
    }
  }
  return result;

}

template<class ClusterRefT>
void
TkStripMeasurementDet::buildSimpleRecHit( const ClusterRefT& cluster,
					  const TrajectoryStateOnSurface& ltp,
					  std::vector<SiStripRecHit2D>& res ) const
{
  const GeomDetUnit& gdu( specificGeomDet());
  VLocalValues vlv = theCPE->localParametersV( *cluster, gdu, ltp);
  for(VLocalValues::const_iterator it=vlv.begin();it!=vlv.end();++it){
    res.push_back(SiStripRecHit2D( it->first, it->second, geomDet().geographicalId(), cluster));
  }
}


void 
TkStripMeasurementDet::simpleRecHits( const TrajectoryStateOnSurface& ts, std::vector<SiStripRecHit2D> &result) const
{
  if (empty || !isActive()) return;

  if(!isRegional){//old implemetation with DetSet
    result.reserve(detSet_.size());
    for ( new_const_iterator ci = detSet_.begin(); ci != detSet_.end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( handle_, ci ); 
      if (accept(cluster))
	buildSimpleRecHit( cluster, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<cluster.key();
    }
  }else{
    result.reserve(endCluster - beginCluster);
    for (const_iterator ci = beginCluster ; ci != endCluster; ci++) {      
      if (isMasked(*ci)) continue;
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(regionalHandle_,ci-regionalHandle_->begin_record());     
      if (accept(clusterRef))
	buildSimpleRecHit( clusterRef, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<geomDet().geographicalId().rawId()<<" key: "<<clusterRef.key();
    }
  }
}


void
TkStripMeasurementDet::set128StripStatus(bool good, int idx) { 
   if (idx == -1) {
       std::fill(bad128Strip_, bad128Strip_+6, !good);
       hasAny128StripBad_ = !good;
   } else {
       bad128Strip_[idx] = !good;
       if (good == false) {
            hasAny128StripBad_ = false;
       } else { // this should not happen, as usually you turn on all fibers
                // and then turn off the bad ones, and not vice-versa,
                // so I don't care if it's not optimized
            hasAny128StripBad_ = true;
            for (int i = 0; i < (totalStrips_ >> 7); i++) {
                if (bad128Strip_[i] == false) hasAny128StripBad_ = false;
            }
       }    
   }
    
}

bool
TkStripMeasurementDet::testStrips(float utraj, float uerr) const {
    int16_t start = (int16_t) std::max<float>(utraj - 3*uerr, 0);
    int16_t end   = (int16_t) std::min<float>(utraj + 3*uerr, totalStrips_);

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
    BSBIT bsbc = badStripBlocks_.begin(), bsbe = badStripBlocks_.end();

    int16_t bad = 0, largestBadBlock = 0;
    for (BSBIT bsbc = badStripBlocks_.begin(), bsbe = badStripBlocks_.end(); bsbc != bsbe; ++bsbc) {
        if (bsbc->last  < start) continue;
        if (bsbc->first > end)   break;
        int16_t thisBad = std::min(bsbc->last, end) - std::max(bsbc->first, start);
        if (thisBad > largestBadBlock) largestBadBlock = thisBad;
        bad += thisBad;
    }

    bool ok = (bad < (end-start)) && 
              (uint16_t(bad) <= badStripCuts_.maxBad) && 
              (uint16_t(largestBadBlock) <= badStripCuts_.maxConsecutiveBad);

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

