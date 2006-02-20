/******* \class RecHit4D *******
 *
 * Base class for 4-parameters recHits.
 *
 *  $Revision: 1.1 $
 *  $Date   17/02/2006 12:02:03 CET $
 *  \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "DataFormats/TrackingRecHit/interface/RecHit4D.h"

/* Collaborating Class Header */

/* C++ Headers */

/* ====================================================================== */
bool RecHit4D::isInitialized(false);
AlgebraicMatrix RecHit4D::theProjectionMatrix;

/* Constructor */ 
RecHit4D::RecHit4D() {
}

/* Destructor */ 
RecHit4D::~RecHit4D() {
}

/* Operations */ 
AlgebraicVector RecHit4D::parameters() const {
  return param( localPosition(), localDirection() );
}

AlgebraicSymMatrix RecHit4D::parametersError() const {
  return parError( localPositionError(), localDirectionError());
}

void RecHit4D::initialize() const {

  theProjectionMatrix = AlgebraicMatrix( 4, 5, 0);
  theProjectionMatrix[0][1] = 1;
  theProjectionMatrix[1][2] = 1;
  theProjectionMatrix[2][3] = 1;
  theProjectionMatrix[3][4] = 1;

  isInitialized = true;
}

AlgebraicVector RecHit4D::param( const LocalPoint& lp, const LocalVector& lv) const {
  AlgebraicVector result(4);
  result[2] = lp.x();
  result[3] = lp.y();
  result[0] = lv.x()/lv.z();
  result[1] = lv.y()/lv.z();    
  return result;
}

/** local Error + AlignmentPositionError if this is set for the DetUnit */
AlgebraicSymMatrix RecHit4D::parError( const LocalError& lp,
                                       const LocalError& lv) const {
  AlgebraicSymMatrix m(4);
  // if ( det().alignmentPositionError()) {
  //   LocalError lape = 
  //     ErrorFrameTransformer().transform( det().alignmentPositionError()->globalError(), 
  //                                        det().surface());
  //   // LocalError lade = 
  //   //   ErrorFrameTransformer().transform( det().alignmentPositionError()->globalError(), 
  //   //                                      det().surface());
  //   m[0][0] = lv.xx();
  //   m[0][1] = lv.xy();
  //   m[1][1] = lv.yy();
  //   m[2][2] = lp.xx()+lape.xx();
  //   m[2][3] = lp.xy()+lape.xy(); 
  //   m[3][3] = lp.yy()+lape.yy();
  // } else {
  m[0][0] = lv.xx();
  m[0][1] = lv.xy();
  m[1][1] = lv.yy();
  m[2][2] = lp.xx();
  m[2][3] = lp.xy(); 
  m[3][3] = lp.yy();
  //};
  return m;
}
