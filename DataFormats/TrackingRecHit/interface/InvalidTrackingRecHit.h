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

  virtual InvalidTrackingRecHit * clone() const override {return new InvalidTrackingRecHit(*this);}
#ifndef __GCCXML__
  virtual RecHitPointer cloneSH() const override { return RecHitPointer(clone());}
#endif

  
  virtual AlgebraicVector parameters() const override;

  virtual AlgebraicSymMatrix parametersError() const override;

  virtual AlgebraicMatrix projectionMatrix() const override;

  virtual int dimension() const override { return 0;}

  virtual LocalPoint localPosition() const override;

  virtual LocalError localPositionError() const override;

  virtual std::vector<const TrackingRecHit*> recHits() const override;

  virtual std::vector<TrackingRecHit*> recHits() override;

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const override;

private:

  void throwError() const;

};

class InvalidTrackingRecHitNoDet final : public InvalidTrackingRecHit {
public:

  InvalidTrackingRecHitNoDet() {}
  InvalidTrackingRecHitNoDet(Surface const & surface, Type type) : InvalidTrackingRecHit(type), m_surface(&surface){}

  virtual InvalidTrackingRecHitNoDet * clone() const override {return new InvalidTrackingRecHitNoDet(*this);}

  const Surface* surface() const override {  return  m_surface; }

 private:
  Surface const * m_surface;

};

#endif
