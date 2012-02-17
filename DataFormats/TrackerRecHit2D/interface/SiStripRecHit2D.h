#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

class SiStripRecHit2D : public  BaseSiTrackerRecHit2DLocalPos{
public:

  SiStripRecHit2D(): BaseSiTrackerRecHit2DLocalPos(),cluster_(),clusterRegional_(),
		     sigmaPitch_(-1.){}

  ~SiStripRecHit2D() {} 

  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRef const&  cluster); 

  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;
  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRegionalRef const& cluster);
  
  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  ClusterRegionalRef const&  cluster_regional()  const { return clusterRegional_;}

  ClusterRef const&  cluster()  const { return cluster_;}

  void setClusterRef(ClusterRef const & ref) { cluster_ = ref; }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { clusterRegional_ = ref; }
  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}

 private:

  // DetSetVector ref
  ClusterRef cluster_;


  // SiStripRefGetter ref.
  ClusterRegionalRef clusterRegional_;

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
