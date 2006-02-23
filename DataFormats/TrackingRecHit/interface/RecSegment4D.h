#ifndef TrackingRecHit_RecSegment4D_h
#define TrackingRecHit_RecSegment4D_h

/** \class RecSegment4D
 *
 * Base class for 4-parameters segments.
 
 * The 4 parameters measured are x,y,dx/dy and dy/dz, so this class represent
 * segment in space.
 * To be used as base class for all 4D Tracking RecSegments.
 *
 * $Date: 22/02/2006 13:03:21 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

/* Collaborating Class Declarations */
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

/* C++ Headers */

/* ====================================================================== */

/* Class RecSegment4D Interface */

class RecSegment4D : public RecSegment {

  public:

/// Destructor
    virtual ~RecSegment4D() {};

/* Operations */ 
    /// return x,y,dx/dy,dy/dz
    virtual AlgebraicVector parameters() const ;

    /// return cov matrix for x,y,dx/dy,dy/dz
    virtual AlgebraicSymMatrix parametersError() const ;

    ///
    virtual AlgebraicMatrix projectionMatrix() const {
      if ( !isInitialized) initialize();
      return theProjectionMatrix;
    }

    /// Return the number of parameters for this RecHit (4)
    virtual int dimension() const { return 4; }

  private:
    static bool isInitialized;

    static AlgebraicMatrix theProjectionMatrix;

    void initialize() const ;

    AlgebraicVector param( const LocalPoint& lp, const LocalVector& lv) const ;

    /** local Error + AlignmentPositionError if this is set for the DetUnit */
    AlgebraicSymMatrix parError( const LocalError& lp, const LocalError& lv) const ;

};
#endif // TrackingRecHit_RecSegment4D_h

