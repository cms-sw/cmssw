#ifndef SiStripRecHit2DLocalPos_H
#define SiStripRecHit2DLocalPos_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripRecHit2DLocalPos : public  BaseSiStripRecHit2DLocalPos{
public:

  SiStripRecHit2DLocalPos(): BaseSiStripRecHit2DLocalPos() {}

  ~SiStripRecHit2DLocalPos() {}

  SiStripRecHit2DLocalPos( const LocalPoint&, const LocalError&,
			 const DetId&, 
			   const std::vector<const SiStripCluster*>& cluster);  

  virtual SiStripRecHit2DLocalPos * clone() const {return new SiStripRecHit2DLocalPos( * this); }

  const std::vector<const SiStripCluster*>& cluster() const { return cluster_;}
  
private:
  std::vector<const SiStripCluster*>   cluster_;

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
