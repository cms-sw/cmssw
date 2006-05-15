#ifndef SiStripRecHit2DMatchedLocalPos_H
#define SiStripRecHit2DMatchedLocalPos_H


#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

class SiStripRecHit2DMatchedLocalPos : public BaseSiTrackerRecHit2DLocalPos{
 public:
  SiStripRecHit2DMatchedLocalPos(): BaseSiTrackerRecHit2DLocalPos(){}
  ~SiStripRecHit2DMatchedLocalPos(){}
  SiStripRecHit2DMatchedLocalPos( const LocalPoint& pos, const LocalError& err, const DetId& id , const SiStripRecHit2DLocalPos* rMono,const SiStripRecHit2DLocalPos* rStereo);
					 
  const SiStripRecHit2DLocalPos *stereoHit() const { return &componentStereo_;}
  const SiStripRecHit2DLocalPos *monoHit() const { return &componentMono_;}
  
  
  virtual SiStripRecHit2DMatchedLocalPos * clone() const {return new SiStripRecHit2DMatchedLocalPos( * this); }

  
 private:
  const SiStripRecHit2DLocalPos componentMono_,componentStereo_;
};


#endif
