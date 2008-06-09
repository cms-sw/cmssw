#ifndef SiTrackerGSMatchedRecHit2D_H
#define SiTrackerGSMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
class SiTrackerGSRecHit2D;

class SiTrackerGSMatchedRecHit2D : public BaseSiTrackerRecHit2DLocalPos{
  
public:
  
  SiTrackerGSMatchedRecHit2D(): BaseSiTrackerRecHit2DLocalPos(),
			 simhitId_(),
			 simtrackId_(),
			 eeId_(),
                         pixelMultiplicityAlpha_(), 
                         pixelMultiplicityBeta_(),
                         isMatched_(), 
                         componentMono_(),
                         componentStereo_() {}
  
  ~SiTrackerGSMatchedRecHit2D() {}
  
  SiTrackerGSMatchedRecHit2D( const LocalPoint&, const LocalError&,
		       const DetId&,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY,
		       const bool isMatched,
		       const SiTrackerGSRecHit2D* rMono, 
		       const SiTrackerGSRecHit2D* rStereo 
		       );  

  SiTrackerGSMatchedRecHit2D( const LocalPoint&, const LocalError&,
		       const DetId&,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY
		       );  

  virtual SiTrackerGSMatchedRecHit2D * clone() const {return new SiTrackerGSMatchedRecHit2D( * this); }
  
  const int& simhitId()    const { return simhitId_;}
  const int& simtrackId()  const { return simtrackId_;}
  const uint32_t& eeId()   const { return eeId_;}
  const int& simMultX()    const { return pixelMultiplicityAlpha_;}
  const int& simMultY()    const { return pixelMultiplicityBeta_;}
  const bool& isMatched()  const { return isMatched_;}
  const SiTrackerGSRecHit2D *monoHit() const { return &componentMono_;}
  const SiTrackerGSRecHit2D *stereoHit() const { return &componentStereo_;}

 
private:
  int const simhitId_;
  int const simtrackId_;
  uint32_t const eeId_;
  int const pixelMultiplicityAlpha_;
  int const pixelMultiplicityBeta_;
  bool isMatched_;
  SiTrackerGSRecHit2D componentMono_;
  SiTrackerGSRecHit2D componentStereo_;
};



#endif
