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


  /**
   * Default constructor, creating a null 3*3 matrix (all values are 0)
   */
  GlobalErrorBase(): theCartesianError(new AlgebraicSymMatrix(3,0)) {}

  /**
   * Constructor.
   * The symmetric matrix stored as a lower triangular matrix
   */
  GlobalErrorBase(T c11, T c21, T c22, T c31, T c32, T c33):
                   theCartesianError(new AlgebraicSymMatrix(3,0)) {
    (*theCartesianError)(1,1)=c11;
    (*theCartesianError)(2,1)=c21;
    (*theCartesianError)(2,2)=c22;
    (*theCartesianError)(3,1)=c31;
    (*theCartesianError)(3,2)=c32;
    (*theCartesianError)(3,3)=c33;
  }
  
  /**
   * Constructor from SymMatrix. The original matrix has to be a 3*3 matrix.
   */
  GlobalErrorBase(const AlgebraicSymMatrix & err) {
    if (err.num_row() == 3)
      theCartesianError = new AlgebraicSymMatrix(err);
    else {
      theCartesianError = 0;
      //throw DetLogicError("Not 3x3 Error Matrix: set pointer to 0");
      throw cms::Exception("DetLogicError")<<"Not 3x3 Error Matrix: set pointer to 0\n";

    }
  }
  
  ~GlobalErrorBase() {}

  T cxx() const {
    return (*theCartesianError)(1,1);
  }
  
  T cyx() const {
    return (*theCartesianError)(2,1);
  }
  
  T cyy() const {
    return (*theCartesianError)(2,2);
  }
  
  T czx() const {
    return (*theCartesianError)(3,1);
  }
  
  T czy() const {
    return (*theCartesianError)(3,2);
  }
  
  T czz() const {
    return (*theCartesianError)(3,3);
  }
  
  /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  AlgebraicSymMatrix matrix() const {
    return *theCartesianError;
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
    return GlobalErrorBase( this->cxx()+err.cxx(), 
                           this->cyx()+err.cyx(),
                           this->cyy()+err.cyy(),
			   this->czx()+err.czx(),
			   this->czy()+err.czy(),
			   this->czz()+err.czz());
  }
  GlobalErrorBase operator- (const GlobalErrorBase& err) const {
    return GlobalErrorBase( this->cxx()-err.cxx(), 
                           this->cyx()-err.cyx(),
                           this->cyy()-err.cyy(),
			   this->czx()-err.czx(),
			   this->czy()-err.czy(),
			   this->czz()-err.czz());
  }
 
private:

  DeepCopyPointer<AlgebraicSymMatrix> theCartesianError;



};

#endif


