#ifndef InvalidTrackingRecHit_H
#define InvalidTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class InvalidTrackingRecHit GCC11_FINAL : public TrackingRecHit {
public:
  typedef TrackingRecHit::Type Type;
  
  //  InvalidTrackingRecHit(DetId id, Type type ) : TrackingRecHit(id, type) , surface_(nullptr) {}
  // InvalidTrackingRecHit(DetId id, GeomDet const * idet, Type type ) : TrackingRecHit(id, idet, type) ,  surface_(idet ? &(det()->surface()) : nullptr) {}
  // InvalidTrackingRecHit(GeomDet const * idet, Type type ) : TrackingRecHit(idet == nullptr ? DetId(0) : idet->geographicalId(), idet, type) , surface_(idet ? &(det()->surface()) : nullptr) {}
  InvalidTrackingRecHit(GeomDet const & idet, Type type ) : TrackingRecHit(idet.geographicalId(), &idet, type) , surface_(&det()->surface()) {}

  InvalidTrackingRecHit() : TrackingRecHit(0, TrackingRecHit::missing) {}

#ifdef NO_DICT
  template<typename ADetLayer>
  InvalidTrackingRecHit(const GeomDet* geom, const ADetLayer * layer, Type type) :
    TrackingRecHit(geom == nullptr ? DetId(0) : geom->geographicalId(), geom, type), 
    surface_(geom ? &(det()->surface()) : ( layer ?  &(layer->surface()) : nullptr))
      {}
#endif

  virtual ~InvalidTrackingRecHit() {}

  const Surface* surface() const GCC11_OVERRIDE {  return  surface_; }
  
  virtual InvalidTrackingRecHit * clone() const GCC11_OVERRIDE {return new InvalidTrackingRecHit(*this);}
  
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

 private:
  Surface const * surface_;

};

#endif
