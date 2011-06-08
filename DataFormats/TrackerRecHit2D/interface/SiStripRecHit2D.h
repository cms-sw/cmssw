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

  typedef OmniClusterRef::ClusterRef ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef;


  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
		   const DetId& id,
		   ClusterRef const& cluster) : 
    cluster_(cluster),
    BaseSiTrackerRecHit2DLocalPos(pos,err,id),
    sigmaPitch_(-1.) {}


  SiStripRecHit2D(const LocalPoint& pos, const LocalError& err,
		  const DetId& id,
		  ClusterRegionalRef const& cluster) :
    cluster_(cluster),
    BaseSiTrackerRecHit2DLocalPos(pos,err,id),
    sigmaPitch_(-1.) {}
						 
  
  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  ClusterRegionalRef cluster_regional()  const { 
    return cluster_.cluster_regional();
  }

  ClusterRef cluster()  const { 
    return cluster_.cluster();
  }

  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}

  
private:


  // DetSetVector ref
  // ClusterRef cluster_;
  // SiStripRefGetter ref.
  //ClusterRegionalRef clusterRegional_;

  // new game
  OmniClusterRef cluster_;

  /// cache for the matcher....
  mutable double sigmaPitch_;  // transient....
 
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
