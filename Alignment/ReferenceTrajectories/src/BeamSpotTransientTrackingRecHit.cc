/** \file BeamSpotTransientTrackingRecHit.cc
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2011/05/18 10:19:48 $
 * by         : $Author: mussgill $
 */

#include "Alignment/ReferenceTrajectories/interface/BeamSpotTransientTrackingRecHit.h"

AlgebraicVector BeamSpotTransientTrackingRecHit::parameters() const
{
  AlgebraicVector result(1);
  result[0] = localPosition().x();
  return result;
}

AlgebraicSymMatrix BeamSpotTransientTrackingRecHit::parametersError() const
{
  LocalError le = localPositionError();
  AlgebraicSymMatrix m(1);
  m[0][0] = le.xx();
  return m;
}

void BeamSpotTransientTrackingRecHit::initialize() const
{
  theProjectionMatrix = AlgebraicMatrix( 1, 5, 0);
  theProjectionMatrix[0][3] = 1;
  
  isInitialized = true;
}

bool BeamSpotTransientTrackingRecHit::isInitialized(false);
AlgebraicMatrix BeamSpotTransientTrackingRecHit::theProjectionMatrix;
