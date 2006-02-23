#ifndef TrackingRecHit_RecSegment2D_h
#define TrackingRecHit_RecSegment2D_h

/** \class RecSegment2D
 *
 * Base class for 2-parameters segments measuring position and direction in X
 * projection.
 *  
 * Implements the AbstractDetMeasurement part of the interface
 * for 2D RecHits in terms of localPosition() and localPositionError() and
 * Direction. This segment is measuring the position and the direction in just
 * one projection, the "X". Typical use case is a segment reconstructed only in
 * X projection.
 * To be used as base class for all 2D positional-directional segments.
 * The coordinate measured is assumend to be the local "x" and "dx/dz"
 *
 * $Date: 22/02/2006 13:00:40 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

/* Collaborating Class Declarations */
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Vector/interface/LocalPoint.h"

/* C++ Headers */

/* ====================================================================== */

/* Class RecSegment2D Interface */

class RecSegment2D : public RecSegment{

  public:

/// Destructor
    virtual ~RecSegment2D() {};

/* Operations */ 
    virtual AlgebraicVector parameters() const {
      return param( localPosition(), localDirection());
    }

    virtual AlgebraicSymMatrix parametersError() const {
      return parError( localPositionError(), localDirectionError());
    }

    /** return the projection matrix, which must project a parameter vector,
     * whose components are (q/p, dx/dz, dy/dz, x, y), into the vector returned
     * by parameters() */
    virtual AlgebraicMatrix projectionMatrix() const {
      if ( !isInitialized) initialize();
      return theProjectionMatrix;
    }
    
    /// return 2
    virtual int dimension() const { return 2;}

  private:
    static bool isInitialized;

    static AlgebraicMatrix theProjectionMatrix;

    void initialize() const {
      isInitialized=true;
      theProjectionMatrix = AlgebraicMatrix( 2, 5, 0);
      theProjectionMatrix[0][1]=1;
      theProjectionMatrix[1][3]=1;
    }

    AlgebraicVector param( const LocalPoint& lp, const LocalVector& lv) const {
      AlgebraicVector result(2);
      result[1]=lp.x();
      result[0]=lv.x()/lv.z();
      return result;
    }

    AlgebraicSymMatrix parError( const LocalError& lp,
                                 const LocalError& lv) const;

};
#endif // TrackingRecHit_RecSegment2D_h

