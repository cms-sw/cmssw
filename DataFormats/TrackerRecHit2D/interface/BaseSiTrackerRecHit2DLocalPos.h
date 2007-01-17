#ifndef BaseSiTrackerRecHit2DLocalPos_H
#define BaseSiTrackerRecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/DetId/interface/DetId.h"

class BaseSiTrackerRecHit2DLocalPos : public RecHit2DLocalPos {
public:

  BaseSiTrackerRecHit2DLocalPos(): id_(0) {}

  ~BaseSiTrackerRecHit2DLocalPos() {}

  BaseSiTrackerRecHit2DLocalPos( const LocalPoint& p, const LocalError&e,
			       const DetId&id) : pos_(p), err_(e), id_(id){}

  //  virtual BaseSiTrackerRecHit2DLocalPos * clone() const {return new BaseSiTrackerRecHit2DLocalPos( * this); }

  virtual LocalPoint localPosition() const {return pos_;}

  virtual LocalError localPositionError() const{ return err_;}

  virtual DetId geographicalId() const {return id_;}
  
 private:
  
  LocalPoint pos_;
  LocalError err_;
  DetId id_;
};

// Comparison operators
inline bool operator<( const BaseSiTrackerRecHit2DLocalPos& one, const BaseSiTrackerRecHit2DLocalPos& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
