#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"

class SiTrackerGSRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
public:
  
  SiTrackerGSRecHit2D(): GSSiTrackerRecHit2DLocalPos(),
			 simhitId_(),
			 simtrackId_(),
			 eeId_(),
			 pixelMultiplicityAlpha_(), 
			 pixelMultiplicityBeta_() {}
  
  ~SiTrackerGSRecHit2D() {}
  
  SiTrackerGSRecHit2D( const LocalPoint&, const LocalError&,
		       const DetId&,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY );  
  
  virtual SiTrackerGSRecHit2D * clone() const {return new SiTrackerGSRecHit2D( * this); }
  
  const int& simhitId()    const { return simhitId_;}
  const int& simtrackId()  const { return simtrackId_;}
  const uint32_t& eeId()    const { return eeId_;}
  const int& simMultX()    const { return pixelMultiplicityAlpha_;}
  const int& simMultY()    const { return pixelMultiplicityBeta_;}
  
private:
  int const simhitId_;
  int const simtrackId_;
  uint32_t const eeId_;
  int const pixelMultiplicityAlpha_;
  int const pixelMultiplicityBeta_;
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
