#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

class SiStripRecHit2D : public  BaseSiTrackerRecHit2DLocalPos{
public:

  SiStripRecHit2D(): BaseSiTrackerRecHit2DLocalPos(),
		     sigmaPitch_(-1.){}

  ~SiStripRecHit2D() {} 

  typedef OmniClusterRef::ClusterRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;


  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
		   const DetId& id,
		   ClusterRef const& cluster) : 
    BaseSiTrackerRecHit2DLocalPos(pos,err,id),
    sigmaPitch_(-1.), cluster_(cluster) {}
 

  SiStripRecHit2D(const LocalPoint& pos, const LocalError& err,
		  const DetId& id,
		  ClusterRegionalRef const& cluster) :
    BaseSiTrackerRecHit2DLocalPos(pos,err,id),
    sigmaPitch_(-1.), cluster_(cluster) {}
						 
  
  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  

  OmniClusterRef const & omniCluster() const { return cluster_;}
  
  
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }
  
  ClusterRef cluster()  const { 
    return cluster_.cluster();
  }

  
  void setClusterRef(ClusterRef const & ref) { cluster_.setRef(ref); }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { cluster_.setRef(ref); }


  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}

  
private:

  /// cache for the matcher....
  mutable double sigmaPitch_;  // transient....

  // DetSetVector ref
  // ClusterRef cluster_;
  // SiStripRefGetter ref.
  //ClusterRegionalRef clusterRegional_;

  // new game
  OmniClusterRef cluster_;

 
};

// Comparison operators
inline bool operator<( const SiStripRecHit2D& one, const SiStripRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
