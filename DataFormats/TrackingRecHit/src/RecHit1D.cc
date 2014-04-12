
/*
 *  See header file for a description of this class.
 *
 *  \author S. Lacaprara, G. Cerminara
 */

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"

// #include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"



// Just the x
AlgebraicVector RecHit1D::parameters() const {
  AlgebraicVector result(1);
  result[0] = localPosition().x();
  return result;
}



// local Error + AlignmentPositionError if this is set for the DetUnit
AlgebraicSymMatrix RecHit1D::parametersError() const {
  LocalError le = localPositionError();
  AlgebraicSymMatrix m(1);
  // FIXME: Remove this dependence from Geometry
//   if ( det().alignmentPositionError()) {
//     LocalError lape = 
//       ErrorFrameTransformer().transform( det().alignmentPositionError()->globalError(), 
// 					 det().surface());
//     m[0][0] = le.xx()+lape.xx();
//   } else {
    m[0][0] = le.xx();
//   }
  return m;
}

// Return an initialized matrix.
static const AlgebraicMatrix initializeMatrix() {
  AlgebraicMatrix matrix( 1, 5, 0);
  matrix[0][3] = 1;
  return matrix;
}

const AlgebraicMatrix RecHit1D::theProjectionMatrix(initializeMatrix());
