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

  if (active_ == false) {
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
    uerr= sqrt(theStripGDU->specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
     if (testStrips(utraj,uerr)) {
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
        result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
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
	//      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster, 
	SiStripClusterRef clusterref = edmNew::makeRefTo( handle_, leftCluster ); 
	TransientTrackingRecHit::RecHitPointer recHit = buildRecHit(clusterref, 
								    stateOnThisDet.localParameters());
	std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
	if ( diffEst.first ) {
	  result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						   diffEst.second));
	}
	else break; // exit loop on first incompatible hit
      }
    }
    
    for ( ; rightCluster != detSet_.end(); rightCluster++) {
      SiStripClusterRef clusterref = edmNew::makeRefTo( handle_, rightCluster ); 
      TransientTrackingRecHit::RecHitPointer recHit = buildRecHit( clusterref, 
								   stateOnThisDet.localParameters());
      std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
      if ( diffEst.first) {
	result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						 diffEst.second));
      }
      else break; // exit loop on first incompatible hit
    }
  }// end block with DetSet
  else{
    const_iterator rightCluster = 
      std::find_if( beginCluster, endCluster, StripClusterAboveU( utraj));

    if ( rightCluster != beginCluster) {
      // there are hits on the left of the utraj
      const_iterator leftCluster = rightCluster;
      while ( --leftCluster >=  beginCluster) {
	//      TransientTrackingRecHit* recHit = buildRecHit( *leftCluster, 
	//std::cout << "=====making ref in fastMeas left " << std::endl;
	SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle_,leftCluster-regionalHandle_->begin_record());
	TransientTrackingRecHit::RecHitPointer recHit = buildRecHit(clusterref, 
								    stateOnThisDet.localParameters());
	std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
	if ( diffEst.first ) {
	  result.push_back( TrajectoryMeasurement( stateOnThisDet, recHit, 
						   diffEst.second));
	}
	else break; // exit loop on first incompatible hit
      }
    }
    
    for ( ; rightCluster != endCluster; rightCluster++) {
      //std::cout << "=====making ref in fastMeas rigth " << std::endl;
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(regionalHandle_,rightCluster-regionalHandle_->begin_record());
      TransientTrackingRecHit::RecHitPointer recHit = buildRecHit( clusterref, 
								   stateOnThisDet.localParameters());
      std::pair<bool,double> diffEst = est.estimate(stateOnThisDet, *recHit);
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
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, but active ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::missing), 0.F));
     } else { 
       //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, and inactive ";
       result.push_back( TrajectoryMeasurement( stateOnThisDet, InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F));
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
  if (active_ == false) return result; // GIO

  if(!isRegional){//old implemetation with DetSet
    for ( new_const_iterator ci = detSet_.begin(); ci != detSet_.end(); ++ ci ) {
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( handle_, ci ); 
      result.push_back( buildRecHit( cluster, ts.localParameters()));
    }
  }else{
    for (const_iterator ci = beginCluster ; ci != endCluster; ci++) {      
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(regionalHandle_,ci-regionalHandle_->begin_record());     
      result.push_back( buildRecHit( clusterRef, ts.localParameters()));
    }
  }
  return result;

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
    int start = (int) (utraj - 3*uerr); if (start < 0) start = 0;
    int end   = (int) (utraj + 3*uerr); if (end > totalStrips_) end = totalStrips_;

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

    int cur = start, curapv = start >> 7, good = 0;
    while (cur < end) {
        int nextapv = (cur & ~(127)) + 128;
        if (bad128Strip_[curapv]) { 
            cur = nextapv; continue;
        }
        int next = std::min(end, nextapv); // all before "next" is good for APVs and fibers 
                                           // [*] next > cur by contract.
        if (bsbc != bsbe) {  // are there any bad strips?
            // skip all blocks to our left
            while (bsbc->last < cur) { bsbc++; if (bsbc == bsbe) break; }
            if ((bsbc != bsbe)) {
                if (bsbc->first <= cur) { // in the block
                    cur = bsbc->last+1; bsbc++;  continue;
                } 
                if (bsbc->first < next) { // there are bad strips before "next"
                    next = bsbc->first;   // so we better stop at the beginning of that block
                    // as we didn't fall in "if (bsbc->first <= cur)" we know
                    // cur < bsbc->first, so [*] is still true
                }
            }
        }
        // because of [*] (next - cur) > 0
        good += next - cur; // all strips up to next-1 are good
        cur  = next;        // now reach for the unknown
   }
   
   /* LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<","<<
        " U = " << utraj << " +/- " << uerr << 
        "; Range [" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
        "= [" << start << "," << end << "]" <<
        " total strips:" << (end-start) << ", good:" << good << ", bad:" << (end-start-good) << 
        ". " << (good >= 1 ? "OK" : "NO"); */

//#define RecoTracker_MeasurementDet_TkStripMeasurementDet_RECOUNT_IN_SLOW_AND_STUPID_BUT_SAFE_WAY
// I can be dangerous to blindly trust some "supposed-to-be-smart" algorithm ...
// ... expecially if I wrote it   (gpetrucc)
#ifdef  RecoTracker_MeasurementDet_TkStripMeasurementDet_RECOUNT_IN_SLOW_AND_STUPID_BUT_SAFE_WAY
    bsbc = badStripBlocks_.begin();
    cur  = start;
    int safegood = 0;
    while (cur < end) {
        if (bad128Strip_[cur >> 7]) { cur++; continue; }
        // skip all blocks to our left
        while ((bsbc != bsbe) && (bsbc->last < cur)) { bsbc++; }
        if ((bsbc != bsbe) && (bsbc->first <= cur)) { cur++; continue; }
        safegood++; cur++;
    }
    //LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<", "<<
    //        " safegood = " << safegood << " while good = " << good <<
    //        "; I am  " << (safegood == good ? "safe" : "STUPID"); // no offense to anyone, of course
#endif // of #ifdef  RecoTracker_MeasurementDet_TkStripMeasurementDet_RECOUNT_IN_SLOW_AND_STUPID_BUT_SAFE_WAY

    return (good >= 1); //to be tuned
}

