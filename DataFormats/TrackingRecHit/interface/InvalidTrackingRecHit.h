#ifndef InvalidTrackingRecHit_H
#define InvalidTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class InvalidTrackingRecHit : public TrackingRecHit {
public:
  typedef TrackingRecHit::Type Type;

  InvalidTrackingRecHit( const DetId& id, Type type ) : detId_(id), type_(type) {}
  InvalidTrackingRecHit() : detId_(0), type_(TrackingRecHit::missing) {}

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
  virtual Type getType() const { return type_; }

private:

  DetId detId_;
  Type type_;

  void throwError() const;

};

#endif
