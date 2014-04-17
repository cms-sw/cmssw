#ifndef TrackerSingleRecHit_H
#define TrackerSingleRecHit_H


#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


/*  a Hit composed by a "single" measurement
 * it has a reference to a cluster and a local position&error
 */
class TrackerSingleRecHit : public BaseTrackerRecHit { 
public:

  typedef BaseTrackerRecHit Base;

  TrackerSingleRecHit(){}
  
  
  typedef OmniClusterRef::ClusterPixelRef ClusterPixelRef;
  typedef OmniClusterRef::ClusterStripRef ClusterStripRef;

  // no position (as in persistent)
  TrackerSingleRecHit(DetId id,
		      OmniClusterRef const&  clus) : 
    Base(id, trackerHitRTTI::single), cluster_(clus){}
  
  template<typename CluRef>
  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      GeomDet const & idet,
		      CluRef const&  clus) : Base(p,e,idet, trackerHitRTTI::single), cluster_(clus){}

  // for projected...
  template<typename CluRef>
  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      GeomDet const & idet, trackerHitRTTI::RTTI rt,
		      CluRef const&  clus) : Base(p,e,idet, rt), cluster_(clus){}


  // a single hit is on a detunit
  const GeomDetUnit* detUnit() const {
    return static_cast<const GeomDetUnit*>(det());
  }

  
  // used by trackMerger (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const  GCC11_FINAL { return cluster_;}

  OmniClusterRef const & omniClusterRef() const { return cluster_;}
  OmniClusterRef const & omniCluster() const { return cluster_;}
  // for rekeying...
  OmniClusterRef & omniClusterRef()  { return cluster_;}
  OmniClusterRef & omniCluster()  { return cluster_;}

  ClusterPixelRef cluster_pixel()  const { 
    return cluster_.cluster_pixel();
  }
  
  ClusterStripRef cluster_strip()  const { 
    return cluster_.cluster_strip();
  }
  
  SiStripCluster const & stripCluster() const { 
    return cluster_.stripCluster();
  }  

  SiPixelCluster const & pixelCluster() const {
    return cluster_.pixelCluster();
  }


  // void setClusterRef(const &  OmniClusterRef ref) {  cluster_ =ref;}
  void setClusterPixelRef(ClusterPixelRef const & ref) {  cluster_ = OmniClusterRef(ref); }
  void setClusterStripRef(ClusterStripRef const & ref) {  cluster_ = OmniClusterRef(ref); }



  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const  GCC11_FINAL;


  bool sharesInput(TrackerSingleRecHit const & other) const {
    return cluster_== other.cluster_;
  }

  bool sameCluster( OmniClusterRef const & oh) const {
    return oh == cluster_;
  }

  virtual std::vector<const TrackingRecHit*> recHits() const;
  virtual std::vector<TrackingRecHit*> recHits();

private:
 
 // new game
  OmniClusterRef cluster_;
 
};

#endif
