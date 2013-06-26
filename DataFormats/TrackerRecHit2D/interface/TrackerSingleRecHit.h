#ifndef TrackerSingleRecHit_H
#define TrackerSingleRecHit_H


#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"


/*  a Hit composed by a "single" measurement
 * it has a reference to a cluster and a local position&error
 */
class TrackerSingleRecHit : public BaseTrackerRecHit { 
public:

  typedef BaseTrackerRecHit Base;

  TrackerSingleRecHit(){}
  
  
  typedef OmniClusterRef::ClusterPixelRef ClusterPixelRef;
  typedef OmniClusterRef::ClusterStripRef ClusterStripRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;


  // no position (as in persistent)
  TrackerSingleRecHit(DetId id,
		      OmniClusterRef const&  clus) : 
    Base(id, trackerHitRTTI::single), cluster_(clus){}
  
  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,
		      OmniClusterRef const&  clus) : Base(p,e,id, trackerHitRTTI::single), cluster_(clus){}

  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,
		      ClusterPixelRef const&  clus) : Base(p,e,id, trackerHitRTTI::single), cluster_(clus){}

  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,
		      ClusterStripRef const&  clus) : Base(p,e,id, trackerHitRTTI::single), cluster_(clus){}

  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,  
		      ClusterRegionalRef const& clus) :  Base(p,e,id, trackerHitRTTI::single), cluster_(clus){}
  
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
  
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }

  SiStripCluster const & stripCluster() const { 
    return cluster_.stripCluster();
  }  

  // void setClusterRef(const &  OmniClusterRef ref) {  cluster_ =ref;}
  void setClusterPixelRef(ClusterPixelRef const & ref) {  cluster_ = OmniClusterRef(ref); }
  void setClusterStripRef(ClusterStripRef const & ref) {  cluster_ = OmniClusterRef(ref); }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { cluster_ = OmniClusterRef(ref); }



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
