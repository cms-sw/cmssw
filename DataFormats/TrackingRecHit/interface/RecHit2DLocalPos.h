#ifndef RecHit2DLocalPos_H
#define RecHit2DLocalPos_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"


class RecHit2DLocalPos : public TrackingRecHit {
public:

  typedef GloballyPositioned<float>::LocalPoint LocalPoint;
  
  RecHit2DLocalPos(DetId id) : TrackingRecHit(id) {}
  RecHit2DLocalPos(TrackingRecHit::id_type id=0) : TrackingRecHit(id) {}
  virtual ~RecHit2DLocalPos() {}
  
  virtual RecHit2DLocalPos * clone() const = 0;
  
  virtual AlgebraicVector parameters() const;

  virtual AlgebraicSymMatrix parametersError() const;

  virtual AlgebraicMatrix projectionMatrix() const {
    if ( !isInitialized) initialize();
    return theProjectionMatrix;
  }

  virtual int dimension() const { return 2;}

  virtual LocalPoint localPosition() const = 0;

  virtual LocalError localPositionError() const = 0;

  virtual std::vector<const TrackingRecHit*> recHits() const;

  virtual std::vector<TrackingRecHit*> recHits();

private:

  static bool isInitialized;

  static AlgebraicMatrix theProjectionMatrix;

  void initialize() const;

};

#endif
