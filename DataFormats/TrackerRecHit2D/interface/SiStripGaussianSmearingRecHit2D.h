#ifndef SiStripGaussianSmearingRecHit2D_H
#define SiStripGaussianSmearingRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class SiStripGaussianSmearingRecHit2D : public BaseSiTrackerRecHit2DLocalPos{

public:
  
  SiStripGaussianSmearingRecHit2D(): BaseSiTrackerRecHit2DLocalPos() {}
  
  ~SiStripGaussianSmearingRecHit2D() {}
  
  SiStripGaussianSmearingRecHit2D( const LocalPoint&, const LocalError&,
				   const DetId&,
				   const unsigned int&,
				   const PSimHit&);  
  
  virtual SiStripGaussianSmearingRecHit2D * clone() const {return new SiStripGaussianSmearingRecHit2D( * this); }
  
  const unsigned int&  simtrackId()  const { return simtrackId_;}
  const PSimHit&       simhit()      const { return simhit_;}
  
private:
  unsigned int const simtrackId_;
  PSimHit      const simhit_;
  
};

// Comparison operators
inline bool operator<( const SiStripGaussianSmearingRecHit2D& one, const SiStripGaussianSmearingRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
