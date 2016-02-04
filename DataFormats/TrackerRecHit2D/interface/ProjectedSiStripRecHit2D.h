#ifndef ProjectedSiStripRecHit2D_H
#define ProjectedSiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class ProjectedSiStripRecHit2D : public BaseSiTrackerRecHit2DLocalPos {
public:

  ProjectedSiStripRecHit2D() {};
  ProjectedSiStripRecHit2D( const LocalPoint& pos, const LocalError& err, const DetId& id , 
			    const SiStripRecHit2D* originalHit) :
    BaseSiTrackerRecHit2DLocalPos(pos, err, id), originalHit_(*originalHit) {}
    
  virtual ProjectedSiStripRecHit2D* clone() const {return new ProjectedSiStripRecHit2D( *this); }

  const SiStripRecHit2D& originalHit() const {return originalHit_;}
  SiStripRecHit2D& originalHit() {return originalHit_;}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const {
    return originalHit().sharesInput(other,what);
  }
  virtual std::vector<const TrackingRecHit*> recHits() const{
    std::vector<const TrackingRecHit*> rechits(1,&originalHit_);
    return rechits;
  }
  virtual std::vector<TrackingRecHit*> recHits() {
    std::vector<TrackingRecHit*> rechits(1,&originalHit_);
    return rechits;
  }


private:

  SiStripRecHit2D originalHit_;

};

#endif
