#ifndef SiStripRecHit2DLocalPos_H
#define SiStripRecHit2DLocalPos_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class SiStripRecHit2DLocalPos : public  BaseSiTrackerRecHit2DLocalPos{
public:

  SiStripRecHit2DLocalPos(): BaseSiTrackerRecHit2DLocalPos(),cluster_() {}

  ~SiStripRecHit2DLocalPos() {}

  SiStripRecHit2DLocalPos( const LocalPoint&, const LocalError&,
			 const DetId&, 
			   edm::Ref< edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  > const&  cluster);  

  virtual SiStripRecHit2DLocalPos * clone() const {return new SiStripRecHit2DLocalPos( * this); }

  edm::Ref<edm::DetSetVector<SiStripCluster> ,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > const&  cluster()  const { return cluster_;}
  
private:
  //  std::vector<const SiStripCluster*>   cluster_;
  edm::Ref<edm::DetSetVector<SiStripCluster>,SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster>  >  const cluster_;

};

// Comparison operators
inline bool operator<( const SiStripRecHit2DLocalPos& one, const SiStripRecHit2DLocalPos& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
