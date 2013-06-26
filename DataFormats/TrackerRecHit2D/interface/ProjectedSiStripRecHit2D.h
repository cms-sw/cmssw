#ifndef ProjectedSiStripRecHit2D_H
#define ProjectedSiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class ProjectedSiStripRecHit2D : public BaseTrackerRecHit {
public:

  typedef BaseTrackerRecHit Base;
  ProjectedSiStripRecHit2D() {};
  ProjectedSiStripRecHit2D( const LocalPoint& pos, const LocalError& err, const DetId& id , 
			    const SiStripRecHit2D* originalHit) :
    BaseTrackerRecHit(pos, err, id, trackerHitRTTI::proj), originalHit_(*originalHit) {}
    
  virtual ProjectedSiStripRecHit2D* clone() const {return new ProjectedSiStripRecHit2D( *this); }

  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }


  // used by trackMerger (to be improved)
  virtual OmniClusterRef const & firstClusterRef() const { return  originalHit().firstClusterRef();}


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
