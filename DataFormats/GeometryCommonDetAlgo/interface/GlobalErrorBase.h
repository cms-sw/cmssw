#ifndef GlobalErrorType_H
#define GlobalErrorType_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
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
  }
  
  /**
   * Constructor from SymMatrix. The original matrix has to be a 3*3 matrix.
   */
  GlobalErrorBase(const AlgebraicSymMatrix & err) {
    if (err.num_row() == 3)
      theCartesianError = asSMatrix<3>(err);
    else {
      //throw DetLogicError("Not 3x3 Error Matrix: set pointer to 0");
      throw cms::Exception("DetLogicError")<<"Not 3x3 Error Matrix: set pointer to 0\n";

    }
  }

   /**
   * Constructor from SymMatrix. The original matrix has to be a 3*3 matrix.
   */
    GlobalErrorBase(const AlgebraicSymMatrix33 & err) : 
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
  
  /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  AlgebraicSymMatrix matrix() const {
    return asHepMatrix(theCartesianError);
  }
 /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  const AlgebraicSymMatrix33 & matrix_new() const {
    return theCartesianError;
  }


  T rerr(const GlobalPoint& aPoint) const {
    T r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if(r2 != 0) 
      return (1./r2)*(x2*cxx() + 2.*xy*cyx() + y2*cyy());
    else 
      return 0.5*(cxx() + cyy());  
  }

  T phierr(const GlobalPoint& aPoint) const {
    T r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if (r2 != 0) 
      return (1./(r2*r2))*(y2*cxx() - 2.*xy*cyx() + x2*cyy());
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

  AlgebraicSymMatrix33 theCartesianError;

};

#endif


