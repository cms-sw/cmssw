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
  if (active_ == false) return result; // GIO

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
                // and then turn off the bad ones, and not vice-versa
            hasAny128StripBad_ = true;
            for (int i = 0; i < (totalStrips_ >> 7); i++) {
                if (bad128Strip_[i] == false) hasAny128StripBad_ = false;
            }
       }    
   }
    
}

#ifndef RecoTracker_MeasurementDet_TkStripMeasurementDet_BADSTRIP_FROM_NOISE
bool
TkStripMeasurementDet::testStrips(float utraj, float uerr) const {
    int start = (int) (utraj - 3*uerr); if (start < 0) start = 0;
    int end   = (int) (utraj + 3*uerr); if (end > totalStrips_) end = totalStrips_;

    if (start >= end) { // which means either end <=0 or start >= totalStrips_
        LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<","<<
            " U = " << utraj << " +/- " << uerr << 
            "; Range [" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
            ": YOU'RE COMPLETELY OFF THE MODULE.";
        return false;
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
   
   LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<","<<
        " U = " << utraj << " +/- " << uerr << 
        "; Range [" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
        "= [" << start << "," << end << "]" <<
        " total strips:" << (end-start) << ", good:" << good << ", bad:" << (end-start-good) << 
        ". " << (good >= 1 ? "OK" : "NO");

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
    LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<", "<<
        " safegood = " << safegood << " while good = " << good <<
        "; I am  " << (safegood == good ? "safe" : "STUPID"); // no offense to anyone, of course
#endif // of #ifdef  RecoTracker_MeasurementDet_TkStripMeasurementDet_RECOUNT_IN_SLOW_AND_STUPID_BUT_SAFE_WAY

    return (good >= 1); //to be tuned
}

#else // of #ifndef RecoTracker_MeasurementDet_TkStripMeasurementDet_BADSTRIP_FROM_NOISE
bool
TkStripMeasurementDet::testStrips(float utraj, float uerr) const {
    int istart = (int) (utraj - 3*uerr); if (istart < 0) istart = 0;
    SiStripNoises::ContainerIterator start = stripNoises_.first + istart;
    int iend = (int) (utraj + 3*uerr); 
    SiStripNoises::ContainerIterator end = stripNoises_.first + iend;
    if (end > stripNoises_.second) end = stripNoises_.second;

    bool goodapv = 0;
    for (int apv = (istart >> 7), apvend = (iend >> 7); // (x >> 7) = x/128
          apv <= apvend; apv++) {
        if (!bad128Strip_[apv]) { goodapv = true; break; }
    }

    if (!goodapv) {
        LogDebug("TkStripMeasurementDet") << "Testing module " << id_ <<", "<<
            " U = " << utraj << " +/- " << (3*uerr) << 
            " Range U:[" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
            "= I:[" << istart << "," << iend << "]" <<
            " NO GOOD APV/FIBER";
        return false;
    }

    int found = 0, off = 0;
    while (start < end) {
        found++;
        if (bad128Strip_[istart / 128] || (*start < 0)) off++;
        start++; istart++;
    }
    LogDebug("TkStripMeasurementDet") << "Testing module " << (geomDet().geographicalId().rawId()) <<", "<<
        " U = " << utraj << " +/- " << (3*uerr) << 
        " Range U:[" << (utraj - 3*uerr) << ", " << (utraj + 3*uerr) << "] " << 
        "= I:[" << istart << "," << iend << "]" <<
        " total strips: " << found << ", bad: " << off << 
        ", verdict: " << ((off == 0) || (off < found));
    return (off == 0) || (off < found); //to be tuned
    // succeed if there either there are no bad strips 
    // (which might mean also that no strip DB is there),
    // or there is at least *one* good strip
}
#endif // of #ifndef RecoTracker_MeasurementDet_TkStripMeasurementDet_BADSTRIP_FROM_NOISE

//  void
//  TkStripMeasurementDet::setNoises(const SiStripNoises::Range noises) 
//  {
//    stripNoises_ = noises;
//  /*// count bad strips
//    int found = 0, off = 0;
//    for (SiStripNoises::ContainerIterator it = stripNoises_.first;
//          it != stripNoises_.second; it++) {
//          found++; if (*it < 0) off++;
//    }
//    //std::cout << "[*GIO*] DetID " << mydetid << ": bad strips = " << off << "/" << found << "\n";
//  */
//  }
