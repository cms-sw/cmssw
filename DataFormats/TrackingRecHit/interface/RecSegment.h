#ifndef TrackingRecHit_RecSegment_h
#define TrackingRecHit_RecSegment_h

/** \class RecSegment
 *
 * Base class for reconstructed segments.
 *  
 * In addition to RecHit, it has direction, chi2 and other stuff
 *
 * $Date: 2007/01/17 23:07:06 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

/* Collaborating Class Declarations */
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

/* C++ Headers */

/* ====================================================================== */

/* Class RecSegment Interface */

class RecSegment : public TrackingRecHit{

 public:
  RecSegment(DetId id) : TrackingRecHit(id) {}
  RecSegment(TrackingRecHit::id_type=0) : TrackingRecHit(id) {}

  /// Destructor
  virtual ~RecSegment() {};

  /// Local direction
  virtual LocalVector localDirection() const = 0;

  /// Error on the local direction
  virtual LocalError localDirectionError() const = 0;

  /// Chi2 of the segment fit
  virtual double chi2() const  = 0 ;

  /// Degrees of freedom of the segment fit
  virtual int degreesOfFreedom() const = 0 ;

  /// Dimension (in parameter space)
  virtual int dimension() const = 0 ;

};
#endif // TrackingRecHit_RecSegment_h

