
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/01/24 14:09:25 $
 *  $Revision: 1.1 $
 *  \author S. Lacaprara, G. Cerminara
 */

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"

// #include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"



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



// Initialize the projection matrix
void RecHit1D::initialize() const {
  theProjectionMatrix = AlgebraicMatrix( 1, 5, 0);
  theProjectionMatrix[0][3] = 1;
  
  isInitialized = true;
}



bool RecHit1D::isInitialized(false);



AlgebraicMatrix RecHit1D::theProjectionMatrix;

