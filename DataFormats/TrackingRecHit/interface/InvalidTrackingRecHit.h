#ifndef InvalidTrackingRecHit_H
#define InvalidTrackingRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class InvalidTrackingRecHit : public TrackingRecHit {
public:
  typedef TrackingRecHit::Type Type;
  
  InvalidTrackingRecHit(GeomDet const & idet, Type type ) : TrackingRecHit(idet, type)  {}
  explicit InvalidTrackingRecHit(Type type) : TrackingRecHit(DetId(0), type) {}

  InvalidTrackingRecHit() : TrackingRecHit(DetId(0), TrackingRecHit::missing) {}

  virtual ~InvalidTrackingRecHit() {}

  virtual InvalidTrackingRecHit * clone() const GCC11_OVERRIDE {return new InvalidTrackingRecHit(*this);}
#ifdef NO_DICT
  virtual RecHitPointer cloneSH() const { return RecHitPointer(clone());}
#endif

  
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

class InvalidTrackingRecHitNoDet GCC11_FINAL : public InvalidTrackingRecHit {
public:

  InvalidTrackingRecHitNoDet() {}
  InvalidTrackingRecHitNoDet(Surface const & surface, Type type) : InvalidTrackingRecHit(type), m_surface(&surface){}

  virtual InvalidTrackingRecHitNoDet * clone() const GCC11_OVERRIDE {return new InvalidTrackingRecHitNoDet(*this);}

  const Surface* surface() const GCC11_OVERRIDE {  return  m_surface; }

 private:
  Surface const * m_surface;

};

#endif
