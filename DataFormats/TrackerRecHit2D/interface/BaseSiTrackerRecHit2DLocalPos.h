#ifndef BaseSiTrackerRecHit2DLocalPos_H
#define BaseSiTrackerRecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/DetId/interface/DetId.h"

class BaseSiTrackerRecHit2DLocalPos : public RecHit2DLocalPos {
public:

  BaseSiTrackerRecHit2DLocalPos(): RecHit2DLocalPos(0) {}

  ~BaseSiTrackerRecHit2DLocalPos() {}

  BaseSiTrackerRecHit2DLocalPos( const LocalPoint& p, const LocalError&e,
				 DetId id) :  RecHit2DLocalPos(id), pos_(p), err_(e){}

  //  virtual BaseSiTrackerRecHit2DLocalPos * clone() const {return new BaseSiTrackerRecHit2DLocalPos( * this); }

  virtual LocalPoint localPosition() const ;

  virtual LocalError localPositionError() const ;

  bool hasPositionAndError() const ; 
 
  virtual void getKfComponents( KfComponentsHolder & holder ) const ; 

  const LocalPoint & localPositionFast()      const { return pos_; }
  const LocalError & localPositionErrorFast() const { return err_; }

 private:

  void throwExceptionUninitialized(const char *where) const;
  
  LocalPoint pos_;
  LocalError err_;
};

// Comparison operators
inline bool operator<( const BaseSiTrackerRecHit2DLocalPos& one, const BaseSiTrackerRecHit2DLocalPos& other) {
  return ( one.geographicalId() < other.geographicalId() );
}

#endif
