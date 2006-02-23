/** \file
 *
 * $Date:  22/02/2006 13:02:21 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "DataFormats/TrackingRecHit/interface/RecSegment2D.h"

/* Collaborating Class Header */

/* C++ Headers */

/* ====================================================================== */

/* static member definition */
bool RecSegment2D::isInitialized( false);
AlgebraicMatrix RecSegment2D::theProjectionMatrix;

/* Operations */ 
AlgebraicSymMatrix RecSegment2D::parError( const LocalError& lp, 
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
