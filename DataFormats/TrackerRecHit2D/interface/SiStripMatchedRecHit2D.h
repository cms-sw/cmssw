#ifndef SiStripMatchedRecHit2D_H
#define SiStripMatchedRecHit2D_H


#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class SiStripMatchedRecHit2D : public BaseSiTrackerRecHit2DLocalPos{
 public:
  SiStripMatchedRecHit2D(): BaseSiTrackerRecHit2DLocalPos(){}
  ~SiStripMatchedRecHit2D(){}
  SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err, const DetId& id , const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo);
					 
  const SiStripRecHit2D *stereoHit() const { return &componentStereo_;}
  const SiStripRecHit2D *monoHit() const { return &componentMono_;}
  
  
  virtual SiStripMatchedRecHit2D * clone() const {return new SiStripMatchedRecHit2D( * this); }
 
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
 
  
 private:
  SiStripRecHit2D componentMono_,componentStereo_;
};


#endif
