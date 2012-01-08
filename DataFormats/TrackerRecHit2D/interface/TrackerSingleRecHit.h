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
  typedef OmniClusterRef::ClusterRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;


  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,
		      ClusterPixelRef const&  clus) : Base(p,e,id), cluster_(clus){}

  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,
		      ClusterRef const&  clus) : Base(p,e,id), cluster_(clus){}

  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e,
		      DetId id,  
		      ClusterRegionalRef const& clus) :  Base(p,e,id), cluster_(clus){}
  
  OmniClusterRef const & omniCluster() cons { return cluster_;}

  ClusterPixelRef cluster_pixel()  const { 
    return cluster_.cluster_pixel();
  }
  
  
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }
  
  ClusterRef cluster_strip()  const { 
    return cluster_.cluster_strip();
  }


  void setClusterRef(ClusterRef const & ref) {  cluster_ = OmniClusterRef(ref); }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { cluster_ = OmniClusterRef(ref); }



  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;


  bool sharesInput(TrackerSingleRecHit const & other) const {
    return cluster_== other.cluster_;
  }


  virtual std::vector<const TrackingRecHit*> recHits() const;
  virtual std::vector<TrackingRecHit*> recHits();

private:
 
 // new game
  OmniClusterRef cluster_;
 
};

#endif
