/** \file BeamSpotTransientTrackingRecHit.cc
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2010/09/10 12:06:43 $
 * by         : $Author: mussgill $
 */

#include "Alignment/ReferenceTrajectories/interface/BeamSpotTransientTrackingRecHit.h"

AlgebraicVector BeamSpotTransientTrackingRecHit::parameters() const {
  AlgebraicVector result(1);
  result[0] = localPosition().x();
  return result;
}

AlgebraicSymMatrix BeamSpotTransientTrackingRecHit::parametersError() const {
  LocalError le = localPositionError();
  AlgebraicSymMatrix m(1);
  m[0][0] = le.xx();
  return m;
}

static AlgebraicMatrix initialize() {
  AlgebraicMatrix ret(1, 5, 0);
  ret[0][3] = 1;
  return ret;
}

const AlgebraicMatrix BeamSpotTransientTrackingRecHit::theProjectionMatrix = initialize();
