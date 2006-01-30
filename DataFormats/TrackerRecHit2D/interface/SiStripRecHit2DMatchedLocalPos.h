#ifndef SiStripRecHit2DMatchedLocalPos_H
#define SiStripRecHit2DMatchedLocalPos_H


#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"

class SiStripRecHit2DMatchedLocalPos : public BaseSiStripRecHit2DLocalPos{
 public:
  SiStripRecHit2DMatchedLocalPos(): BaseSiStripRecHit2DLocalPos(){}
  ~SiStripRecHit2DMatchedLocalPos(){}
  SiStripRecHit2DMatchedLocalPos( const LocalPoint& pos, const LocalError& err, const DetId& id , const SiStripRecHit2DLocalPos* rMono,const SiStripRecHit2DLocalPos* rStereo);
					 
  const SiStripRecHit2DLocalPos *stereoHit() const { return componentStereo_;}
  const SiStripRecHit2DLocalPos *monoHit() const { return componentMono_;}
  
  
  virtual SiStripRecHit2DMatchedLocalPos * clone() const {return new SiStripRecHit2DMatchedLocalPos( * this); }

  
 private:
  const SiStripRecHit2DLocalPos *componentStereo_,*componentMono_;
};


#endif
