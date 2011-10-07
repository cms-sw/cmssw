#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

#include "float.h"

class SiStripRecHit1D : public RecHit1D{
public:

 SiStripRecHit1D(): RecHit1D(),cluster_(),clusterRegional_(),
    sigmaPitch_(-1.){}

  ~SiStripRecHit1D() {} 

  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  SiStripRecHit1D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRef const&  cluster); 

  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;
  SiStripRecHit1D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   ClusterRegionalRef const& cluster);
  
  /// method to facilitate the convesion from 2D to 1D hits
  SiStripRecHit1D(const SiStripRecHit2D*);

  virtual SiStripRecHit1D * clone() const {return new SiStripRecHit1D( * this); }
  
  ClusterRegionalRef const&  cluster_regional()  const { return clusterRegional_;}

  ClusterRef const&  cluster()  const { return cluster_;}

  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 

  virtual LocalPoint localPosition() const;

  virtual LocalError localPositionError() const;

  bool hasPositionAndError() const ; 

  

  void setClusterRef(ClusterRef const & ref) { cluster_ = ref; }
  void setClusterRegionalRef(ClusterRegionalRef const & ref) { clusterRegional_ = ref; }
  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}


  virtual std::vector<const TrackingRecHit*> recHits() const;
  virtual std::vector<TrackingRecHit*> recHits();


 private:
  void throwExceptionUninitialized(const char *where) const;
  
  LocalPoint pos_;
  LocalError err_;

  // DetSetVector ref
  ClusterRef cluster_;

  // SiStripRefGetter ref.
  ClusterRegionalRef clusterRegional_;

  /// cache for the matcher....
  mutable double sigmaPitch_;  // transient....
 
};

// Comparison operators
inline bool operator<( const SiStripRecHit1D& one, const SiStripRecHit1D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
