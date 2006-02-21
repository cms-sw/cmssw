#ifndef SiPixelRecHit_H
#define SiPixelRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class SiPixelRecHit : public  BaseSiStripRecHit2DLocalPos {
public:

  SiPixelRecHit(): BaseSiStripRecHit2DLocalPos () {}

  ~SiPixelRecHit() {}

  SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 const SiPixelCluster * cluster);  

  virtual SiPixelRecHit * clone() const {return new SiPixelRecHit( * this); }

  const SiPixelCluster * cluster() const { return cluster_;}
  
private:
  const SiPixelCluster * cluster_;

};

// Comparison operators
inline bool operator<( const SiPixelRecHit& one, const SiPixelRecHit& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
