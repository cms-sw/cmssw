#ifndef RecHit2DLocalPos_H
#define RecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

class RecHit2DLocalPos : public TrackingRecHit {
public:
  typedef GloballyPositioned<float>::LocalPoint LocalPoint;

  RecHit2DLocalPos(DetId id) : TrackingRecHit(id) {}
  RecHit2DLocalPos(TrackingRecHit::id_type id = 0) : TrackingRecHit(id) {}
  ~RecHit2DLocalPos() override {}

  RecHit2DLocalPos* clone() const override = 0;

  AlgebraicVector parameters() const override;

  AlgebraicSymMatrix parametersError() const override;

  AlgebraicMatrix projectionMatrix() const override { return theProjectionMatrix; }

  int dimension() const override { return 2; }

  LocalPoint localPosition() const override = 0;

  LocalError localPositionError() const override = 0;

  std::vector<const TrackingRecHit*> recHits() const override;

  std::vector<TrackingRecHit*> recHits() override;

private:
  static const AlgebraicMatrix theProjectionMatrix;
};

#endif
