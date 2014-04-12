#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"

AlgebraicVector RecHit2DLocalPos::parameters() const 
{
  AlgebraicVector result(2);
  LocalPoint lp = localPosition();
  result[0] = lp.x();
  result[1] = lp.y();
  return result;
}


/** local Error + AlignmentPositionError if this is set for the DetUnit */
AlgebraicSymMatrix RecHit2DLocalPos::parametersError() const {
  AlgebraicSymMatrix m(2);
  LocalError le( localPositionError());
  m[0][0] = le.xx();
  m[0][1] = le.xy();
  m[1][1] = le.yy();
  return m;
}

std::vector<const TrackingRecHit*> RecHit2DLocalPos::recHits() const {
  std::vector<const TrackingRecHit*> nullvector;
  return nullvector; 
}
std::vector<TrackingRecHit*> RecHit2DLocalPos::recHits() {
  std::vector<TrackingRecHit*> nullvector;
  return nullvector; 
}

// static member definition
static const AlgebraicMatrix initializeMatrix()
{
  AlgebraicMatrix aMatrix( 2, 5, 0);
  aMatrix[0][3] = 1;
  aMatrix[1][4] = 1;
  return aMatrix;
}
const AlgebraicMatrix RecHit2DLocalPos::theProjectionMatrix{initializeMatrix()};
