#ifndef BaseSiStripRecHit2DLocalPos_H
#define BaseSiStripRecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class BaseSiStripRecHit2DLocalPos : public RecHit2DLocalPos {
public:

  BaseSiStripRecHit2DLocalPos(): id_(0) {}

  ~BaseSiStripRecHit2DLocalPos() {}

  BaseSiStripRecHit2DLocalPos( const LocalPoint& p, const LocalError&e,
			       const DetId&id) : pos_(p), err_(e), id_(id){}

  //  virtual BaseSiStripRecHit2DLocalPos * clone() const {return new BaseSiStripRecHit2DLocalPos( * this); }

  virtual LocalPoint localPosition() const {return pos_;}

  virtual LocalError localPositionError() const{ return err_;}

  virtual DetId geographicalId() const {return id_;}
  
 private:
  
  LocalPoint pos_;
  LocalError err_;
  DetId id_;
};

// Comparison operators
inline bool operator<( const BaseSiStripRecHit2DLocalPos& one, const BaseSiStripRecHit2DLocalPos& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
