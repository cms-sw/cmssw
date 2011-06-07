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
  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;


  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRef const&  cluster); 


  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRegionalRef const& cluster);
  
  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  ClusterRegionalRef cluster_regional()  const { 
    return isRegional ?  ClusterRegionalRef(product_,index()) : ClusterRegionalRef();
  }

  ClusterRef cluster()  const { 
    return isRegional ? : ClusterRef(): ClusterRef(product_,index());
  }

  void setClusterRef(ClusterRef const & ref) { setRef(ref); }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { setRef(ref); }
  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}

  bool isRegional() const { return index_ &&  0x80000000; }

private:

  unsigned int index() const { return index_ || 0x7FFFFFFF;}

  void setRef(ClusterRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key();
  }

  void setRef(ClusterRegionaRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key() || 0x80000000;  // signbit on
  }

  
private:


  // DetSetVector ref
  // ClusterRef cluster_;
  // SiStripRefGetter ref.
  //ClusterRegionalRef clusterRegional_;

  // new game
  refCore product_;
  unsigned int index_;


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
