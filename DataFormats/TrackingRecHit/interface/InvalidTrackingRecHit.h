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

  ~InvalidTrackingRecHit() override {}

  InvalidTrackingRecHit * clone() const override {return new InvalidTrackingRecHit(*this);}
#ifndef __GCCXML__
  RecHitPointer cloneSH() const override { return RecHitPointer(clone());}
#endif

  
  AlgebraicVector parameters() const override;

  AlgebraicSymMatrix parametersError() const override;

  AlgebraicMatrix projectionMatrix() const override;

  int dimension() const override { return 0;}

  LocalPoint localPosition() const override;

  LocalError localPositionError() const override;

  std::vector<const TrackingRecHit*> recHits() const override;

  std::vector<TrackingRecHit*> recHits() override;

  bool sharesInput( const TrackingRecHit* other, SharedInputType what) const override;

private:

  void throwError() const;

};

class InvalidTrackingRecHitNoDet final : public InvalidTrackingRecHit {
public:

  InvalidTrackingRecHitNoDet() {}
  InvalidTrackingRecHitNoDet(Surface const & surface, Type type) : InvalidTrackingRecHit(type), m_surface(&surface){}

  InvalidTrackingRecHitNoDet * clone() const override {return new InvalidTrackingRecHitNoDet(*this);}

  const Surface* surface() const override {  return  m_surface; }

 private:
  Surface const * m_surface;

};

#endif
