#ifndef GlobalErrorType_H
#define GlobalErrorType_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"

  /**
   * Templated class representing a symmetric 3*3 matrix describing,  according
   * to the ErrorWeightType tag, a (cartesian) covariance matrix or the weight
   * matrix (the inverse of the covariance matrix).
   * \li To have a covariance matrix, the ErrorMatrixTag has to be used, and a 
   * typedef is available as GlobalError
   * \li To have a weight matrix, the WeightMatrixTag has to be used, and a 
   * typedef is available as Globalweight
   * 
   * The typedefs should be used in the code.
   */


template <class T, class ErrorWeightType> 
class GlobalErrorBase
{

public:
  /// Tag to request a null error matrix
  class NullMatrix{};

  /**
   * Default constructor, creating a null 3*3 matrix (all values are 0)
   */
  GlobalErrorBase() {}

  /** 
   * Obsolete  Constructor that allocates a null GlobalErrorBase (it does not create the error matrix at all)
   */
  GlobalErrorBase(const NullMatrix &) {}


  /**
   * Constructor.
   * The symmetric matrix stored as a lower triangular matrix
   */
  GlobalErrorBase(T c11, T c21, T c22, T c31, T c32, T c33) {
    theCartesianError(0,0)=c11;
    theCartesianError(1,0)=c21;
    theCartesianError(1,1)=c22;
    theCartesianError(2,0)=c31;
    theCartesianError(2,1)=c32;
    theCartesianError(2,2)=c33;
    theCartesianError(3,0) = 0.;
    theCartesianError(3,1) = 0.;
    theCartesianError(3,2) = 0.;
    theCartesianError(3,3) = 0.;
  }
  
  /**
   * Constructor.
   * The symmetric matrix stored as a lower triangular matrix (4D)
   */
  GlobalErrorBase(T c11, T c21, T c22, T c31, T c32, T c33, T c41, T c42, T c43, T c44) {
    theCartesianError(0,0)=c11;
    theCartesianError(1,0)=c21;
    theCartesianError(1,1)=c22;
    theCartesianError(2,0)=c31;
    theCartesianError(2,1)=c32;
    theCartesianError(2,2)=c33;
    theCartesianError(3,0)=c41;
    theCartesianError(3,1)=c42;
    theCartesianError(3,2)=c43;
    theCartesianError(3,3)=c44;
  }
  
  /**
   * Constructor from SymMatrix. The original matrix has to be a 3*3 matrix.
   */
 GlobalErrorBase(const AlgebraicSymMatrix33 & err) { 
     theCartesianError(0,0) = err(0,0);
     theCartesianError(1,0) = err(1,0);
     theCartesianError(1,1) = err(1,1);
     theCartesianError(2,0) = err(2,0);
     theCartesianError(2,1) = err(2,1);
     theCartesianError(2,2) = err(2,2);
     theCartesianError(3,0) = 0.;
     theCartesianError(3,1) = 0.;
     theCartesianError(3,2) = 0.;
     theCartesianError(3,3) = 0.;
  }

  /**
   * Constructor from SymMatrix. The original matrix has to be a 4*4 matrix.
   */
 GlobalErrorBase(const AlgebraicSymMatrix44 & err) : 
  theCartesianError(err) { }

  
  ~GlobalErrorBase() {}

  T cxx() const {
    return theCartesianError(0,0);
  }
  
  T cyx() const {
    return theCartesianError(1,0);
  }
  
  T cyy() const {
    return theCartesianError(1,1);
  }
  
  T czx() const {
    return theCartesianError(2,0);
  }
  
  T czy() const {
    return theCartesianError(2,1);
  }
  
  T czz() const {
    return theCartesianError(2,2);
  }

  T ctx() const {
    return theCartesianError(3,0);
  }
  
  T cty() const {
    return theCartesianError(3,1);
  }
  
  T ctz() const {
    return theCartesianError(3,2);
  }

  T ctt() const {
    return theCartesianError(3,3);
  }
  
  /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  const AlgebraicSymMatrix33 matrix() const {
    AlgebraicSymMatrix33 out;
    out(0,0) = theCartesianError(0,0);
    out(1,0) = theCartesianError(1,0);
    out(1,1) = theCartesianError(1,1);
    out(2,0) = theCartesianError(2,0);
    out(2,1) = theCartesianError(2,1);
    out(2,2) = theCartesianError(2,2);
    return out;
  }  

 /**
   * Access method to the matrix,
   * /return The SymMatrix 4x4
   */
  const AlgebraicSymMatrix44& matrix4D() const {
    return theCartesianError;
  }

  T rerr(const GlobalPoint& aPoint) const {
    T r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if(r2 != 0) 
      return std::max<T>(0, (1./r2)*(x2*cxx() + 2.*xy*cyx() + y2*cyy()));
    else 
      return 0.5*(cxx() + cyy());  
  }

  T phierr(const GlobalPoint& aPoint) const {
    T r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if (r2 != 0) 
      return std::max<T>(0, (1./(r2*r2))*(y2*cxx() - 2.*xy*cyx() + x2*cyy()));
    else
      return 0;
  }

  GlobalErrorBase operator+ (const GlobalErrorBase& err) const {
    return GlobalErrorBase(theCartesianError + err.theCartesianError);
  }
  GlobalErrorBase operator- (const GlobalErrorBase& err) const {
    return GlobalErrorBase(theCartesianError - err.theCartesianError);
  }
 
private:

  AlgebraicSymMatrix44 theCartesianError;

};

#endif


