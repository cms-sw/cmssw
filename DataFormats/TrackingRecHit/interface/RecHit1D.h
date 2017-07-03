#ifndef TrackingRecHit_RecHit1D_H
#define TrackingRecHit_RecHit1D_H

/** \class RecHit1D
 *
 * Base class for 1-dimensional recHits
 *  
 *
 * To be used as base class for all 1D positional TrackingRecHits.
 * The coordinate measured is assumend to be the local "x"
 *
 *  \author S. Lacaprara, G. Cerminara
 */

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


class RecHit1D : public TrackingRecHit {
 public:

  RecHit1D(DetId id) : TrackingRecHit(id) {}
  RecHit1D(TrackingRecHit::id_type id=0) : TrackingRecHit(id) {}

  /// Destructor
  ~RecHit1D() override {}


  /// Return just the x
  AlgebraicVector parameters() const override;


  /// Return just "(sigma_x)^2"
  AlgebraicSymMatrix parametersError() const override;


  ///Return the projection matrix
  AlgebraicMatrix projectionMatrix() const override {
    return theProjectionMatrix;
  }

  /// Return the RecHit dimension
  int dimension() const override {
    return 1;
  }


  /// Local position
  LocalPoint localPosition() const override = 0;


  /// Error on the local position
  LocalError localPositionError() const override = 0;


 private:
  static const AlgebraicMatrix theProjectionMatrix;
};
#endif
