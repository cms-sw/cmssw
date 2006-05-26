#ifndef MuonTransientTrackingRecHit_H
#define MuonTransientTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"


class MuonTransientTrackingRecHit: public GenericTransientTrackingRecHit{
public:

  MuonTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit * rh) :
    GenericTransientTrackingRecHit(geom,rh) {
  }
  MuonTransientTrackingRecHit( const MuonTransientTrackingRecHit & other ) :
    GenericTransientTrackingRecHit(other.det(), other.hit()) {
  }

  virtual LocalVector localDirection() const;

  virtual GlobalVector globalDirection() const;

   /// Error on the local direction
  virtual LocalError localDirectionError() const;

   /// Error on the global direction
  virtual GlobalError globalDirectionError() const;
 
  virtual double chi2() const;

  virtual int degreesOfFreedom() const;

  /// assert if this rec hit is a DT rec hit 
  bool isDT() const;

  /// assert if this rec hit is a CSC rec hit 
  bool isCSC() const;

  //   /// assert if this rec hit is a RPC rec hit
  //   bool isRPC() const;
    
private:
   
};
#endif

