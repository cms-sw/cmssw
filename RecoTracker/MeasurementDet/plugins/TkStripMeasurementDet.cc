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
#include<cassert>

TkStripMeasurementDet::TkStripMeasurementDet( const GeomDet* gdet, StMeasurementConditionSet & conditions ) : 
  MeasurementDet (gdet), index_(-1), theDetConditions(&conditions)
  {
    if (dynamic_cast<const StripGeomDetUnit*>(gdet) == 0) {
      throw MeasurementDetException( "TkStripMeasurementDet constructed with a GeomDet which is not a StripGeomDetUnit");
    }
  }

TkStripMeasurementDet::RecHitContainer 
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data) const
{
  RecHitContainer result;
  if ( (!isActive(data)) || isEmpty(data.stripData())) return result;
  if(!isRegional()){//old implemetation with DetSet
    const detset & detSet = data.stripData().detSet(index()); 
    result.reserve(detSet.size());
    for ( new_const_iterator ci = detSet.begin(); ci != detSet.end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( data.stripData().handle(), ci ); 
      if (accept(cluster, data.stripClustersToSkip()))
	result.push_back( buildRecHit( cluster, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<rawId()<<" key: "<<cluster.key();
    }
  }else{
    unsigned int ci = beginClusterI(data.stripData()), ce = endClusterI(data.stripData()); 
    result.reserve(ce-ci);
    for (; ci != ce; ++ci){
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),ci);
      if (isMasked(*clusterRef)) continue;
      if (accept(clusterRef, data.stripClustersToSkip()))
	result.push_back( buildRecHit( clusterRef, ts));
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterRef.key();
      }
  }
  return result;

}


bool
TkStripMeasurementDet::recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator& est, const MeasurementTrackerEvent & data, 
				RecHitContainer & result, std::vector<float> & diffs ) const {
  if ( (!isActive(data)) || isEmpty(data.stripData())) return false;

  auto oldSize = result.size();

  float utraj =  specificGeomDet().specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  if(!isRegional()){//old implemetation with DetSet
 
    const detset & detSet = data.stripData().detSet(index()); 
    auto rightCluster = 
      std::find_if( detSet.begin(), detSet.end(), [utraj](const SiStripCluster& hit) { return hit.barycenter() > utraj; });
    
    if ( rightCluster != detSet.begin()) {
      // there are hits on the left of the utraj
      auto leftCluster = rightCluster;
      while ( --leftCluster >=  detSet.begin()) {
	SiStripClusterRef clusterref = edmNew::makeRefTo( data.stripData().handle(), leftCluster ); 
	bool isCompatible = filteredRecHits(clusterref, stateOnThisDet, est, data.stripClustersToSkip(), result, diffs);
	if(!isCompatible) break; // exit loop on first incompatible hit
      }
    }
    for ( ; rightCluster != detSet.end(); rightCluster++) {
      SiStripClusterRef clusterref = edmNew::makeRefTo( data.stripData().handle(), rightCluster ); 
      bool isCompatible = filteredRecHits(clusterref, stateOnThisDet, est, data.stripClustersToSkip(), result,diffs);
      if(!isCompatible) break; // exit loop on first incompatible hit
    }
    
  }// end block with DetSet
  else{
    
    LogDebug("TkStripMeasurementDet")<<" finding left/ right";
    unsigned int rightCluster = beginClusterI(data.stripData());
    unsigned int endCluster = endClusterI(data.stripData());
    result.reserve(endCluster-rightCluster);
    for (; rightCluster!= endCluster;++rightCluster){
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),rightCluster);
      if (clusterref->barycenter() > utraj) break;
    }
    
    unsigned int leftCluster = 1;
    for (unsigned int iReadBackWard=1; iReadBackWard<=(rightCluster-beginClusterI(data.stripData())) ; ++iReadBackWard){
      leftCluster=rightCluster-iReadBackWard;
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),leftCluster);
      bool isCompatible = filteredRecHits(  clusterref, stateOnThisDet, est, data.stripClustersToSkip(), result,diffs);
	if(!isCompatible) break; // exit loop on first incompatible hit
    }
    for ( ; rightCluster != endCluster; ++rightCluster) {
      SiStripRegionalClusterRef clusterref = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),rightCluster);
      bool isCompatible = filteredRecHits(  clusterref, stateOnThisDet, est, data.stripClustersToSkip(), result,diffs);
      if(!isCompatible) break; // exit loop on first incompatible hit
    }
  }
  return result.size()>oldSize;
}

bool TkStripMeasurementDet::measurements( const TrajectoryStateOnSurface& stateOnThisDet,
					  const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
					  TempMeasurements & result) const {

  if (!isActive(data)) {
    LogDebug("TkStripMeasurementDet")<<" found an inactive module "<<rawId();
    result.add(InvalidTransientRecHit::build(&geomDet(), TrackingRecHit::inactive), 0.F);
    return true;
  }
  
  if (!isEmpty(data.stripData())){
    LogDebug("TkStripMeasurementDet")<<" found hit on this module "<<rawId();
    RecHitContainer rechits;
    std::vector<float>  diffs;
    if (recHits(stateOnThisDet,est,data,result.hits,result.distances)) return true;
  }


  // create a TrajectoryMeasurement with an invalid RecHit and zero estimate

  if (!stateOnThisDet.hasError()) {
    result.add(InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F);
    return false;
  }

  float utraj =  specificGeomDet().specificTopology().measurementPosition( stateOnThisDet.localPosition()).x();
  float uerr= sqrt(specificGeomDet().specificTopology().measurementError(stateOnThisDet.localPosition(),stateOnThisDet.localError().positionError()).uu());
  if (testStrips(utraj,uerr)) {
    //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, but active ";
    result.add(InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::missing), 0.F);
    return false;
  }

  //LogDebug("TkStripMeasurementDet") << " DetID " << id_ << " empty after search, and inactive ";
  result.add(InvalidTransientRecHit::build(&fastGeomDet(), TrackingRecHit::inactive), 0.F);
  return true;

}







void 
TkStripMeasurementDet::simpleRecHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data, std::vector<SiStripRecHit2D> &result) const
{
  if (isEmpty(data.stripData()) || !isActive(data)) return;

  if(!isRegional()){//old implemetation with DetSet
    const detset & detSet = data.stripData().detSet(index()); 
    result.reserve(detSet.size());
    for ( new_const_iterator ci = detSet.begin(); ci != detSet.end(); ++ ci ) {
      if (isMasked(*ci)) continue;
      // for ( ClusterIterator ci=theClusterRange.first; ci != theClusterRange.second; ci++) {
      SiStripClusterRef  cluster = edmNew::makeRefTo( data.stripData().handle(), ci ); 
      if (accept(cluster, data.stripClustersToSkip()))
	buildSimpleRecHit( cluster, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this str from last iteration on"<<rawId()<<" key: "<<cluster.key();
    }
  }else{
    unsigned int ci = beginClusterI(data.stripData()), ce = endClusterI(data.stripData());
    result.reserve(ce-ci);
    for (;ci != ce;++ci){
      SiStripRegionalClusterRef clusterRef = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),ci);
      if (isMasked(*clusterRef)) continue;
      if (accept(clusterRef, data.stripClustersToSkip()))
	buildSimpleRecHit( clusterRef, ts,result);
      else LogDebug("TkStripMeasurementDet")<<"skipping this reg str from last iteration on"<<rawId()<<" key: "<<clusterRef.key();
    }
  }
}



std::tuple<TkStripRecHitIter,TkStripRecHitIter>
TkStripMeasurementDet::hitRange( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data) const
{
  if (isEmpty(data.stripData()) || !isActive(data)) return std::tuple<TkStripRecHitIter,TkStripRecHitIter>();
   if(!isRegional()){//old implemetation with DetSet
     const detset & detSet = data.stripData().detSet(index()); 
     return std::make_tuple(TkStripRecHitIter(detSet.begin(),detSet.end(),*this,ts,data),
			    TkStripRecHitIter(detSet.end(),detSet.end(),*this,ts,data)
			    );
   } else {
     unsigned int ci = beginClusterI(data.stripData()), ce = endClusterI(data.stripData());
     return std::make_tuple(TkStripRecHitIter(ci,ce,*this,ts,data),
			    TkStripRecHitIter(ce,ce,*this,ts,data)
			    );
 
   }
}

void TkStripMeasurementDet::advance(TkStripRecHitIter & hi ) const {
  if(!isRegional()){
    while (!hi.empty()) {
      auto ci = hi.clusterI;
      auto const & data = *hi.data;
      if (isMasked(*ci)) continue;
      SiStripClusterRef  cluster = edmNew::makeRefTo( data.stripData().handle(), ci ); 
      if (accept(cluster, data.stripClustersToSkip())) return;
      ++hi.clusterI;
    }
  } else {
    while (!hi.empty()) {
      auto ci = hi.clusterIreg;
      auto const & data = *hi.data;
      SiStripRegionalClusterRef cluster = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),ci);
      if (isMasked(*cluster)) continue;
      if (accept(cluster, data.stripClustersToSkip())) return;
      ++hi.clusterIreg;
    }
  }
}

SiStripRecHit2D TkStripMeasurementDet::hit(TkStripRecHitIter const & hi ) const {
  const GeomDetUnit& gdu( specificGeomDet());
  auto ci = hi.clusterI;
  auto ciR = hi.clusterIreg;
  auto const & data = *hi.data;
  auto const & ltp = *hi.tsos;
  
  if(!isRegional()){
    SiStripClusterRef  cluster = edmNew::makeRefTo( data.stripData().handle(), ci ); 
    LocalValues lv = cpe()->localParameters( *cluster, gdu, ltp);
    return SiStripRecHit2D(lv.first,lv.second, TSiStripRecHit2DLocalPos::sigmaPitch(lv.first, lv.second,gdu), rawId(), cluster);
  } else {
    SiStripRegionalClusterRef cluster = edm::makeRefToLazyGetter(data.stripData().regionalHandle(),ciR);
    LocalValues lv = cpe()->localParameters( *cluster, gdu, ltp);
    return SiStripRecHit2D(lv.first,lv.second, TSiStripRecHit2DLocalPos::sigmaPitch(lv.first, lv.second,gdu), rawId(), cluster);
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

