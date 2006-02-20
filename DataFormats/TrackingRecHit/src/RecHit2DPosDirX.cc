/******* \class RecHit2DPosDirX *******
 *
 * Base class for 2-parameters recHits measuring position and direction in X
 * projection.
 *
 * $date   20/02/2006 18:15:05 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "DataFormats/TrackingRecHit/interface/RecHit2DPosDirX.h"

/* Collaborating Class Header */

/* C++ Headers */

/* ====================================================================== */

/* static member definition */
bool RecHit2DPosDirX::isInitialized( false);
AlgebraicMatrix RecHit2DPosDirX::theProjectionMatrix;

/* Operations */ 
AlgebraicSymMatrix RecHit2DPosDirX::parError( const LocalError& lp, 
                                              const LocalError& lv) const {
  AlgebraicSymMatrix m(2);
  /// mat[0][0]=sigma (dx/dz)
  /// mat[1][1]=sigma (x)
  /// mat[0][1]=cov(dx/dz,x)
  // if ( det().alignmentPositionError()) {
  //   LocalError lape = 
  //     ErrorFrameTransformer().transform( det().alignmentPositionError()->globalError(), 
  //                                        det().surface());
  //   m[0][0] = lv.xx();
  //   m[0][1] = 0.;
  //   m[1][1] = lp.xx()+lape.xx();
  // } else {
    m[0][0] = lv.xx();
    m[0][1] = 0.;
    m[1][1] = lp.xx();
  //};
  return m;

}


