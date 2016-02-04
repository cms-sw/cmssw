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
 
  // Non const variants needed for cluster re-keying 
  SiStripRecHit2D *stereoHit() { return &componentStereo_;}
  SiStripRecHit2D *monoHit() { return &componentMono_;}
  
  virtual SiStripMatchedRecHit2D * clone() const;
 
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

  virtual std::vector<const TrackingRecHit*> recHits() const; 

  virtual std::vector<TrackingRecHit*> recHits(); 

    
 private:
  SiStripRecHit2D componentMono_,componentStereo_;
};


#endif
