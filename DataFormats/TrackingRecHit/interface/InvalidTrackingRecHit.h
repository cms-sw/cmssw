#ifndef InvalidTrackingRecHit_H
#define InvalidTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class InvalidTrackingRecHit : public TrackingRecHit {
public:

  InvalidTrackingRecHit( const DetId& id) : detId_(id) {}
  InvalidTrackingRecHit() : detId_(0) {}

  virtual ~InvalidTrackingRecHit() {}
  
  virtual InvalidTrackingRecHit * clone() const {return new InvalidTrackingRecHit(*this);}
  
  virtual AlgebraicVector parameters() const;

  virtual AlgebraicSymMatrix parametersError() const;

  virtual AlgebraicMatrix projectionMatrix() const;

  virtual int dimension() const;

  virtual DetId geographicalId() const {return detId_;}

  virtual LocalPoint localPosition() const;

  virtual LocalError localPositionError() const;

  virtual std::vector<const TrackingRecHit*> recHits() const;

  virtual std::vector<TrackingRecHit*> recHits();

  virtual bool isValid() const {return false;}

private:

  DetId detId_;

  void throwError() const;

};

#endif
