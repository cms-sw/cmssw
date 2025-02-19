#ifndef GSSiTrackerRecHit2DLocalPos_H
#define GSSiTrackerRecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/DetId/interface/DetId.h"

class GSSiTrackerRecHit2DLocalPos : public RecHit2DLocalPos {
public:

  GSSiTrackerRecHit2DLocalPos(): RecHit2DLocalPos(0) {}

  ~GSSiTrackerRecHit2DLocalPos() {}

  GSSiTrackerRecHit2DLocalPos( const LocalPoint& p, const LocalError&e,
				 DetId id) :  RecHit2DLocalPos(id), pos_(p), err_(e){}

  //  virtual GSSiTrackerRecHit2DLocalPos * clone() const {return new GSSiTrackerRecHit2DLocalPos( * this); }

  virtual LocalPoint localPosition() const {return pos_;}

  virtual LocalError localPositionError() const{ return err_;}
  
  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const {return false;}
 private:
  
  LocalPoint pos_;
  LocalError err_;
};

// Comparison operators
inline bool operator<( const GSSiTrackerRecHit2DLocalPos& one, const GSSiTrackerRecHit2DLocalPos& other) {
  return ( one.geographicalId() < other.geographicalId() );
}

#endif
