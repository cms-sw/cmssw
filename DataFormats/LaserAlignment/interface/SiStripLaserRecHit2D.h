
#ifndef SiStripLaserRecHit2D_H
#define SiStripLaserRecHit2D_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/DetId/interface/DetId.h"

///
///
///
class SiStripLaserRecHit2D : public RecHit2DLocalPos {

public:

  SiStripLaserRecHit2D(): RecHit2DLocalPos(0) {}
  ~SiStripLaserRecHit2D() {}
  SiStripLaserRecHit2D( const LocalPoint& p, const LocalError& e, const DetId& id ) : RecHit2DLocalPos( id ), position( p ), error( e ) {}

  virtual LocalPoint localPosition() const { return position; }
  virtual LocalError localPositionError() const { return error; }
  virtual SiStripLaserRecHit2D * clone() const { return new SiStripLaserRecHit2D( * this); }

  DetId getDetId() const { return detId; };
  void setDetId( DetId aDetId ) { detId = aDetId; }

 private:
  
  LocalPoint position;
  LocalError error;
  DetId detId;

};



///
/// Comparison operators
///
inline bool operator<( const SiStripLaserRecHit2D& one, const SiStripLaserRecHit2D& other ) {
  return( one.geographicalId() < other.geographicalId() );
}


#endif
