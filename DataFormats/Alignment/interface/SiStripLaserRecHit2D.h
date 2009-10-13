
#ifndef SiStripLaserRecHit2D_H
#define SiStripLaserRecHit2D_H

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

///
///
///
class SiStripLaserRecHit2D : public RecHit2DLocalPos {

public:

  SiStripLaserRecHit2D(): RecHit2DLocalPos(0) {}
  ~SiStripLaserRecHit2D() {}
  SiStripLaserRecHit2D( const LocalPoint& p, const LocalError& e, const SiStripDetId& id ) : RecHit2DLocalPos( id ), position( p ), error( e ) { detId = id; }

  virtual LocalPoint localPosition() const { return position; }
  virtual LocalError localPositionError() const { return error; }
  virtual SiStripLaserRecHit2D* clone() const { return new SiStripLaserRecHit2D( *this ); }

  const SiStripDetId& getDetId( void ) const { return detId; }

 private:
  
  LocalPoint position;
  LocalError error;
  SiStripDetId detId;

};



///
/// Comparison operators
///
inline bool operator<( const SiStripLaserRecHit2D& one, const SiStripLaserRecHit2D& other ) {
  return( one.geographicalId() < other.geographicalId() );
}


#endif
