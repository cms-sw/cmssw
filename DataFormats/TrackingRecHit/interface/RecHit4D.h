#ifndef RECHIT4D_H
#define RECHIT4D_H

/** \class RecHit4D
 *
 * Base class for 4-parameters recHits.
 
 * The 4 parameters measured are x,y,dx/dy and dy/dz, so this class represent
 * segment in space.
 * To be used as base class for all 4D TrackingRecHits.
 *
 * $date   16/02/2006 18:05:07 CET $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

/* Collaborating Class Declarations */
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

/* C++ Headers */

/* ====================================================================== */

/* Class RecHit4D Interface */

class RecHit4D : public TrackingRecHit{

  public:

/** Constructor */ 
    RecHit4D() ;

/** Destructor */ 
    virtual ~RecHit4D() ;

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

    /// Local direction
    virtual LocalVector localDirection() const = 0;

    /// Error on the local direction
    virtual LocalError localDirectionError() const = 0;
  private:

    static bool isInitialized;

    static AlgebraicMatrix theProjectionMatrix;

    void initialize() const ;

    AlgebraicVector param( const LocalPoint& lp, const LocalVector& lv) const ;

    /** local Error + AlignmentPositionError if this is set for the DetUnit */
    AlgebraicSymMatrix parError( const LocalError& lp, const LocalError& lv) const ;
};
#endif // RECHIT4D_H

