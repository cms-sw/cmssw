#ifndef InvalidTrackingRecHit_H
#define InvalidTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class InvalidTrackingRecHit GCC11_FINAL : public TrackingRecHit {
public:
  typedef TrackingRecHit::Type Type;

  InvalidTrackingRecHit(DetId id, Type type ) : TrackingRecHit(id, type) {}
  InvalidTrackingRecHit(DetId id, GeomDet const * idet, Type type ) : TrackingRecHit(id, idet, type) {}
  InvalidTrackingRecHit() : TrackingRecHit(0, TrackingRecHit::missing) {}

  virtual ~InvalidTrackingRecHit() {}
  
  virtual InvalidTrackingRecHit * clone() const {return new InvalidTrackingRecHit(*this);}
  
  virtual AlgebraicVector parameters() const;

  virtual AlgebraicSymMatrix parametersError() const;

  virtual AlgebraicMatrix projectionMatrix() const;

  virtual int dimension() const { return 0;}

  virtual LocalPoint localPosition() const;

  virtual LocalError localPositionError() const;

  virtual std::vector<const TrackingRecHit*> recHits() const;

  virtual std::vector<TrackingRecHit*> recHits();

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

private:

  void throwError() const;

};

#endif
