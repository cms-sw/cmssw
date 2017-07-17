#ifndef GlobalErrorExtendedType_H
#define GlobalErrorExtendedType_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"

template <class T, class ErrorWeightType> 
class GlobalErrorBaseExtended
{

public:
  /// Tag to request a null error matrix
  class NullMatrix{};

  GlobalErrorBaseExtended() {}
  GlobalErrorBaseExtended(const NullMatrix &) {}


  /**
   * Constructor.
   * The symmetric matrix stored as a lower triangular matrix
   */
  GlobalErrorBaseExtended(T c11, T c21, T c31, T c41, T c51, T c61,
                          T c22, T c32, T c42, T c52, T c62,
                          T c33, T c43, T c53, T c63,
                          T c44, T c54, T c64,
                          T c55, T c65,
                          T c66
                          ) {
    theCartesianError(0,0)=c11;
    theCartesianError(1,0)=c21;
    theCartesianError(2,0)=c31;
    theCartesianError(3,0)=c41;
    theCartesianError(4,0)=c51;
    theCartesianError(5,0)=c61;

    theCartesianError(1,1)=c22;
    theCartesianError(2,1)=c32;
    theCartesianError(3,1)=c42;
    theCartesianError(4,1)=c52;
    theCartesianError(5,1)=c62;

    theCartesianError(2,2)=c33;
    theCartesianError(3,2)=c43;
    theCartesianError(4,2)=c53;
    theCartesianError(5,2)=c63;

    theCartesianError(3,3)=c44;
    theCartesianError(4,3)=c54;
    theCartesianError(5,3)=c64;

    theCartesianError(4,4)=c55;
    theCartesianError(5,4)=c65;

    theCartesianError(5,5)=c66;
  }
  
    GlobalErrorBaseExtended(const AlgebraicSymMatrix66 & err) : 
      theCartesianError(err) { }

    GlobalErrorBaseExtended(const AlgebraicSymMatrix33 & err) {
     theCartesianError(0,0)=err[0][0];
     theCartesianError(1,0)=err[1][0];
     theCartesianError(2,0)=err[2][0];
     theCartesianError(3,0)=0;
     theCartesianError(4,0)=0;
     theCartesianError(5,0)=0;

     theCartesianError(1,1)=err[1][1];
     theCartesianError(2,1)=err[2][1];
     theCartesianError(3,1)=0;
     theCartesianError(4,1)=0;
     theCartesianError(5,1)=0;

     theCartesianError(2,2)=err[2][2];
     theCartesianError(3,2)=0;
     theCartesianError(4,2)=0;
     theCartesianError(5,2)=0;

     theCartesianError(3,3)=0;
     theCartesianError(4,3)=0;
     theCartesianError(5,3)=0;

     theCartesianError(4,4)=0;
     theCartesianError(5,4)=0;

     theCartesianError(5,5)=0;
    }
 
  ~GlobalErrorBaseExtended() {}

  T cxx() const {
    return theCartesianError(0,0);
  }
  
  T cyx() const {
    return theCartesianError(1,0);
  }

  T czx() const {
    return theCartesianError(2,0);
  }

  T cphixx() const {
    return theCartesianError(3,0);
  }

  T cphiyx() const {
    return theCartesianError(4,0);
  }

  T cphizx() const {
    return theCartesianError(5,0);
  } 

  T cyy() const {
    return theCartesianError(1,1);
  }

  T czy() const {
    return theCartesianError(2,1);
  }

  T cphixy() const {
    return theCartesianError(3,1);
  }

  T cphiyy() const {
    return theCartesianError(4,1);
  }

  T cphizy() const {
    return theCartesianError(5,1);
  }

  T czz() const {
    return theCartesianError(2,2);
  } 
  
  T cphixz() const {
    return theCartesianError(3,2);
  }

  T cphiyz() const {
    return theCartesianError(4,2);
  } 
    
  T cphizz() const {
    return theCartesianError(5,2);
  }

  T cphixphix() const {
    return theCartesianError(3,3);
  }

  T cphiyphix() const {
    return theCartesianError(4,3);
  }

  T cphizphix() const {
    return theCartesianError(5,3);
  }

  T cphiyphiy() const {
    return theCartesianError(4,4);
  }

  T cphizphiy() const {
    return theCartesianError(5,4);
  }

  T cphizphiz() const {
    return theCartesianError(5,5);
  }

 /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  const AlgebraicSymMatrix66 & matrix() const {
    return theCartesianError;
  }
  const AlgebraicSymMatrix66 & matrix_new() const {
    return theCartesianError;
  }

  //FIXME to be reimplemented
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

  //FIXME to be reimplemented
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

  GlobalErrorBaseExtended operator+ (const GlobalErrorBaseExtended& err) const {
    return GlobalErrorBaseExtended(theCartesianError + err.theCartesianError);
  }
  GlobalErrorBaseExtended operator- (const GlobalErrorBaseExtended& err) const {
    return GlobalErrorBaseExtended(theCartesianError - err.theCartesianError);
  }
 
private:

  AlgebraicSymMatrix66 theCartesianError;

};

#endif
