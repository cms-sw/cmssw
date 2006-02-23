#ifndef TrackingRecHit_RecSegment_h
#define TrackingRecHit_RecSegment_h

/** \class RecSegment
 *
 * Base class for reconstructed segments.
 *  
 * In addition to RecHit, it has direction, chi2 and other stuff
 *
 * $Date: 22/02/2006 13:36:03 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

/* Collaborating Class Declarations */
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalVector.h"

/* C++ Headers */

/* ====================================================================== */

/* Class RecSegment Interface */

class RecSegment : public TrackingRecHit{

  public:

/// Destructor
    virtual ~RecSegment() {};

/* Operations */ 
    /// Local direction
    virtual LocalVector localDirection() const = 0;

    /// Error on the local direction
    virtual LocalError localDirectionError() const = 0;

    virtual double chi2() const  = 0 ;

    virtual int degreesOfFreedom() const = 0 ;


};
#endif // TrackingRecHit_RecSegment_h

