#include "CommonTools/RecoAlgos/interface/ClusterStorer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
// FastSim hits:
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastProjectedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastMatchedTrackerRecHit.h"


#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

namespace helper {

  // -------------------------------------------------------------
  //FIXME (push cluster pointers...)
  void ClusterStorer::addCluster(TrackingRecHitCollection &hits, size_t index)
  {
    TrackingRecHit &newHit = hits[index];
    const std::type_info &hit_type = typeid(newHit);
    if (hit_type == typeid(SiPixelRecHit)) {
      //std::cout << "|  It is a Pixel hit !!" << std::endl;
      pixelClusterRecords_.push_back(PixelClusterHitRecord(static_cast<SiPixelRecHit&>(newHit),
							   hits, index));
    } else if (hit_type == typeid(SiStripRecHit1D)) {
      //std::cout << "|   It is a SiStripRecHit1D hit !!" << std::endl;
      stripClusterRecords_.push_back(StripClusterHitRecord(static_cast<SiStripRecHit1D&>(newHit),
							   hits, index));
    } else if (hit_type == typeid(SiStripRecHit2D)) {
      //std::cout << "|   It is a SiStripRecHit2D hit !!" << std::endl;
      stripClusterRecords_.push_back(StripClusterHitRecord(static_cast<SiStripRecHit2D&>(newHit),
							   hits, index));
    } else if (hit_type == typeid(SiStripMatchedRecHit2D)) {      
      //std::cout << "|   It is a SiStripMatchedRecHit2D hit !!" << std::endl;
      SiStripMatchedRecHit2D &mhit = static_cast<SiStripMatchedRecHit2D&>(newHit);
      stripClusterRecords_.push_back(StripClusterHitRecord(mhit.monoHit(), hits, index));
      stripClusterRecords_.push_back(StripClusterHitRecord(mhit.stereoHit(), hits, index));
    } else if (hit_type == typeid(ProjectedSiStripRecHit2D)) {
      //std::cout << "|   It is a ProjectedSiStripRecHit2D hit !!" << std::endl;
      ProjectedSiStripRecHit2D &phit = static_cast<ProjectedSiStripRecHit2D&>(newHit);
      stripClusterRecords_.push_back(StripClusterHitRecord(phit.originalHit(), hits, index));
    } else {
      if (hit_type == typeid(FastTrackerRecHit)
 	  || hit_type == typeid(FastProjectedTrackerRecHit)
	  || hit_type == typeid(FastMatchedTrackerRecHit)) {
	//std::cout << "|   It is a " << hit_type.name() << " hit !!" << std::endl;
	// FastSim hits: Do nothing instead of caring about FastSim clusters, 
	//               not even sure whether these really exist.
	//               At least predecessor code in TrackSelector and MuonSelector
	//               did not treat them.
      } else {
	// through for unknown types
	throw cms::Exception("UnknownHitType") << "helper::ClusterStorer::addCluster: "
					       << "Unknown hit type " << hit_type.name()
					       << ".\n";
      }
    } // end 'switch' on hit type
    
  }
  
  // -------------------------------------------------------------
  void ClusterStorer::clear()
  {
    pixelClusterRecords_.clear();
    stripClusterRecords_.clear();
  }
  
  // -------------------------------------------------------------
  void ClusterStorer::
  processAllClusters(edmNew::DetSetVector<SiPixelCluster> &pixelDsvToFill,
		     edm::RefProd<edmNew::DetSetVector<SiPixelCluster> > refPixelClusters,
		     edmNew::DetSetVector<SiStripCluster> &stripDsvToFill,
		     edm::RefProd<edmNew::DetSetVector<SiStripCluster> > refStripClusters)
  {
    if (!pixelClusterRecords_.empty()) {
      this->processClusters<SiPixelRecHit, SiPixelCluster>
	(pixelClusterRecords_, pixelDsvToFill, refPixelClusters);
    }
    if (!stripClusterRecords_.empty()) {
      // All we need from the HitType 'SiStripRecHit2D' is the
      // typedef for 'SiStripRecHit2D::ClusterRef'.
      // The fact that rekey<SiStripRecHit2D> is called is irrelevant since
      // ClusterHitRecord<typename SiStripRecHit2D::ClusterRef>::rekey<RecHitType>
      // is specialised such that 'RecHitType' is not used...
      this->processClusters<SiStripRecHit2D, SiStripCluster>
	(stripClusterRecords_, stripDsvToFill, refStripClusters);
    }
  }
  
  //-------------------------------------------------------------
  template<typename HitType, typename ClusterType>
  void ClusterStorer::
  processClusters(std::vector<ClusterHitRecord<typename HitType::ClusterRef> > &clusterRecords,
		  edmNew::DetSetVector<ClusterType>                            &dsvToFill,
		  edm::RefProd< edmNew::DetSetVector<ClusterType> >            &refprod)
  {
    std::sort(clusterRecords.begin(), clusterRecords.end()); // this sorts them by detid 
    typedef
      typename std::vector<ClusterHitRecord<typename HitType::ClusterRef> >::const_iterator
      RIT;
    RIT it = clusterRecords.begin(), end = clusterRecords.end();
    size_t clusters = 0;
    while (it != end) {
      RIT it2 = it;
      uint32_t detid = it->detid();
      
      // first isolate all clusters on the same detid
      while ( (it2 != end) && (it2->detid() == detid)) {  ++it2; }
      // now [it, it2] bracket one detid
      
      // then prepare to copy the clusters
      typename edmNew::DetSetVector<ClusterType>::FastFiller filler(dsvToFill, detid);
      typename HitType::ClusterRef lastRef, newRef;
      for ( ; it != it2; ++it) { // loop on the detid
	// first check if we need to clone the hit
	if (it->clusterRef() != lastRef) { 
	  lastRef = it->clusterRef();
	  // clone cluster
	  filler.push_back( *lastRef );  
	  // make new ref
	  newRef = typename HitType::ClusterRef( refprod, clusters++ );
	} 
	it->template rekey<HitType>(newRef);
      } // end of the loop on a single detid
      
    } // end of the loop on all clusters
  }
  
  //-------------------------------------------------------------
  // helper classes  
  //-------------------------------------------------------------
  // FIXME (migrate to new RTTI and interface)
  // generic rekey (in practise for pixel only...)
  template<typename ClusterRefType> // template for class
  template<typename RecHitType>     // template for member function
  void ClusterStorer::ClusterHitRecord<ClusterRefType>::
  rekey(const ClusterRefType &newRef) const
  {
    TrackingRecHit & genericHit = (*hits_)[index_]; 
    RecHitType *hit = 0;
    if (genericHit.geographicalId().rawId() == detid_) { // a hit on this det, so it's simple
      hit = dynamic_cast<RecHitType *>(&genericHit); //static_cast<RecHitType *>(&genericHit);
    }
    assert (hit != 0);
    assert (hit->cluster() == ref_); // otherwise something went wrong
    hit->setClusterRef(newRef);
  }
  
  // -------------------------------------------------------------
  // Specific rekey for class template ClusterRefType = SiStripRecHit2D::ClusterRef,
  // RecHitType is not used.
  template<>
  template<typename RecHitType> // or template<> to specialise also here?
  void ClusterStorer::ClusterHitRecord<SiStripRecHit2D::ClusterRef>::
  //  rekey<SiStripRecHit2D>(const SiStripRecHit2D::ClusterRef &newRef) const
  rekey(const SiStripRecHit2D::ClusterRef &newRef) const
  {
    TrackingRecHit &genericHit = (*hits_)[index_];
    const std::type_info &hit_type = typeid(genericHit);

    OmniClusterRef * cluRef=0;
    if (typeid(SiStripRecHit1D) == hit_type) {
      cluRef = &static_cast<SiStripRecHit1D&>(genericHit).omniCluster();
     } else if (typeid(SiStripRecHit2D) == hit_type) {
      cluRef = &static_cast<SiStripRecHit2D&>(genericHit).omniCluster();
    } else if (typeid(SiStripMatchedRecHit2D) == hit_type) {
	SiStripMatchedRecHit2D &mhit = static_cast<SiStripMatchedRecHit2D&>(genericHit);
	 cluRef = (SiStripDetId(detid_).stereo() ? &mhit.stereoClusterRef() : &mhit.monoClusterRef());
    } else if (typeid(ProjectedSiStripRecHit2D) == hit_type) {
      cluRef = &static_cast<ProjectedSiStripRecHit2D&>(genericHit).originalHit().omniCluster();
    }
  
    assert(cluRef != 0); // to catch missing RecHit types
    assert(cluRef->key() == ref_.key()); // otherwise something went wrong
    (*cluRef) = OmniClusterRef(newRef);
  }
  
} // end namespace 'helper'
