#ifndef GSSiTrackerRecHit2DLocalPos_H
#define GSSiTrackerRecHit2DLocalPos_H

#include "BaseTrackerRecHit.h"

class GSSiTrackerRecHit2DLocalPos : public BaseTrackerRecHit {
public:

  GSSiTrackerRecHit2DLocalPos() {}

  ~GSSiTrackerRecHit2DLocalPos() {}

  GSSiTrackerRecHit2DLocalPos( const LocalPoint& p, const LocalError&e, GeomDet const & idet) :  
  BaseTrackerRecHit(p,e,idet, trackerHitRTTI::gs) {}

  //  virtual GSSiTrackerRecHit2DLocalPos * clone() const {return new GSSiTrackerRecHit2DLocalPos( * this); }

  
  virtual void getKfComponents( KfComponentsHolder & holder ) const {
     getKfComponents2D(holder);
  }

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const {return false;}

  virtual int dimension() const { return 2;}

  virtual std::vector<const TrackingRecHit*> recHits() const { return std::vector<TrackingRecHit const*>();}
  virtual std::vector<TrackingRecHit*> recHits()  { return std::vector<TrackingRecHit*>();}

  // shall I support FakeCluster?
  virtual OmniClusterRef const & firstClusterRef() const;


};

// Comparison operators
inline bool operator<( const GSSiTrackerRecHit2DLocalPos& one, const GSSiTrackerRecHit2DLocalPos& other) {
  return ( one.geographicalId() < other.geographicalId() );
}

#endif
