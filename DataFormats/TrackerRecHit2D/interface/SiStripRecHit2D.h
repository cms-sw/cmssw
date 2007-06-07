#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"

class SiStripRecHit2D : public  BaseSiTrackerRecHit2DLocalPos{
public:

  SiStripRecHit2D(): BaseSiTrackerRecHit2DLocalPos(),clusterRegional_(),clusterDSV_() {}

  ~SiStripRecHit2D() {} 

  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   edm::Ref< edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  > const&  cluster); 

  SiStripRecHit2D( const LocalPoint&, const LocalError&,
		   const DetId&, 
		   edm::SiStripRefGetter<SiStripCluster>::self_ref const& ); 
  
  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  edm::SiStripRefGetter<SiStripCluster>::self_ref const&  cluster_regional()  const { return clusterRegional_;}

  edm::Ref<edm::DetSetVector<SiStripCluster> ,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > const&  cluster()  const { return clusterDSV_;}
  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
 private:

  // SiStripRefGetter ref.
  edm::SiStripRefGetter<SiStripCluster>::self_ref const clusterRegional_;

  // DetSetVector ref
  edm::Ref<edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  >  const clusterDSV_;
 
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
