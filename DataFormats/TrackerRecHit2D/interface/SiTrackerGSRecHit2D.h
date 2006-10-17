#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class SiTrackerGSRecHit2D : public BaseSiTrackerRecHit2DLocalPos{
  
public:
  
  SiTrackerGSRecHit2D(): BaseSiTrackerRecHit2DLocalPos() {}
  
  ~SiTrackerGSRecHit2D() {}
  
  SiTrackerGSRecHit2D( const LocalPoint&, const LocalError&,
		       const DetId&,
		       const unsigned int&,
		       const PSimHit&,
		       const unsigned int& pixelMultiplicityX,
		       const unsigned int& pixelMultiplicityY );  
  
  virtual SiTrackerGSRecHit2D * clone() const {return new SiTrackerGSRecHit2D( * this); }
  
  const unsigned int&  simtrackId()  const { return simtrackId_;}
  const PSimHit&       simhit()      const { return simhit_;}
  const unsigned int&  simMultX()    const { return pixelMultiplicityAlpha_;}
  const unsigned int&  simMultY()    const { return pixelMultiplicityBeta_;}
  
private:
  unsigned int const simtrackId_;
  PSimHit      const simhit_;
  unsigned int const pixelMultiplicityAlpha_;
  unsigned int const pixelMultiplicityBeta_;
  
};

// Comparison operators
inline bool operator<( const SiTrackerGSRecHit2D& one, const SiTrackerGSRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
