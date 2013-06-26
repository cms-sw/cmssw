#ifndef TrackingRecHit_DTRecSegment2D_h
#define TrackingRecHit_DTRecSegment2D_h

/** \class DTRecSegment2D
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
 * 2D Segments for the muon barrel system.
 * 2D means that this segment has information about position and direction in
 * one projection (r-phi or r-theta/zeta).
 *
 * $Date: 2012/04/30 08:32:03 $
 * $Revision: 1.18 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

/* Collaborating Class Declarations */
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
/* C++ Headers */
#include <iosfwd>

/* Fwd declaration */
class DTSegmentUpdator;

/* ====================================================================== */

/* Class DTRecSegment2D Interface */

class DTRecSegment2D : public RecSegment{

 public:

  /// Constructor
  /// empty c'tor needed by POOL (I guess)
  DTRecSegment2D(): theChi2(0.0), theT0(0.), theVdrift(0.) {}
  
  /// c'tor from hits
  DTRecSegment2D(DetId id, const std::vector<DTRecHit1D>& hits) ;
  
  /// complete constructor
  DTRecSegment2D(DetId id, 
		 LocalPoint &position, LocalVector &direction,
		 AlgebraicSymMatrix & covMatrix, double chi2, 
		 std::vector<DTRecHit1D> &hits1D);

  /// Destructor
  virtual ~DTRecSegment2D();

  /* Operations */ 

  virtual DTRecSegment2D* clone() const { return new DTRecSegment2D(*this);}


  /// the vector of parameters (dx/dz,x)
  virtual AlgebraicVector parameters() const {
    return param( localPosition(), localDirection());
  }

  // The parameter error matrix 
  virtual AlgebraicSymMatrix parametersError() const;

  /** return the projection matrix, which must project a parameter vector,
   * whose components are (q/p, dx/dz, dy/dz, x, y), into the vector returned
   * by parameters() */
  virtual AlgebraicMatrix projectionMatrix() const {
    if ( !isInitialized) initialize();
    return theProjectionMatrix;
  }
    
  /// return 2. The dimension of the matrix
  virtual int dimension() const { return 2;}
    
  /// local position in SL frame
  virtual LocalPoint localPosition() const {return thePosition; }
  
  /// local position error in SL frame
  virtual LocalError localPositionError() const ;
  
  /// the local direction in SL frame
  virtual LocalVector localDirection() const { return theDirection; }

  /// the local direction error (xx,xy,yy) in SL frame: only xx is not 0.
  virtual LocalError localDirectionError() const;

  /// the chi2 of the fit
  virtual double chi2() const { return theChi2; }
    
  /// return the DOF of the segment 
  virtual int degreesOfFreedom() const ;

  // Access to component RecHits (if any)
  virtual std::vector<const TrackingRecHit*> recHits() const ;

  // Non-const access to component RecHits (if any)
  virtual std::vector<TrackingRecHit*> recHits() ;

  /// Access to specific components
  std::vector<DTRecHit1D> specificRecHits() const ;
  
  /// the Covariance Matrix 
  AlgebraicSymMatrix covMatrix() const {return theCovMatrix;}

  /// Get the segment t0 (if recomputed, 0 is returned otherwise)
  double t0() const {return theT0;}
  bool ist0Valid() const {return (theT0 > -998.) ? true : false;}

  /// Get the vDirft as computed by the algo for the computation of the segment t0
  /// (if recomputed, 0 is returned otherwise)
  double vDrift() const {return theVdrift;}

 protected:
  friend class DTSegmentUpdator;
  void setPosition(const LocalPoint& pos);
  void setDirection(const LocalVector& dir);
  void setCovMatrix(const AlgebraicSymMatrix& cov);
  void setChi2(const double& chi2);
  void update(std::vector<DTRecHit1D> & updatedRecHits);
  void setT0(const double& t0);
  void setVdrift(const double& vdrift);

  LocalPoint  thePosition;  // in SL frame
  LocalVector theDirection; // in SL frame
  
  /// mat[0][0]=sigma (dx/dz)
  /// mat[1][1]=sigma (x)
  /// mat[0][1]=cov(dx/dz,x)
  AlgebraicSymMatrix theCovMatrix; // the covariance matrix

  double theChi2;           // chi2 of the fit
  double theT0;             // T0 as coming from the fit
  double theVdrift;             // vDrift as coming from the fit

  std::vector<DTRecHit1D> theHits; // the hits with defined R/L
  

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

};
std::ostream& operator<<(std::ostream& os, const DTRecSegment2D& seg);
#endif // TrackingRecHit_DTRecSegment2D_h

