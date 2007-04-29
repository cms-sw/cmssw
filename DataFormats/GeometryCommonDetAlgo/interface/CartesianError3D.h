#ifndef _TRACKER_CARTESIAN_ERROR_3D_H_
#define _TRACKER_CARTESIAN_ERROR_3D_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"

/** Obsolete. Use GlobalError instead.
 */

template <class T> class CartesianError3D {

public:

  CartesianError3D(): theCartesianError(new AlgebraicSymMatrix(3,0)) {}

  // CartesianError3D is stored as a lower triangular matrix
  CartesianError3D(T c11, T c21, T c22, T c31, T c32, T c33):
                   theCartesianError(new AlgebraicSymMatrix(3,0)) {
    (*theCartesianError)(1,1)=c11;
    (*theCartesianError)(2,1)=c21;
    (*theCartesianError)(2,2)=c22;
    (*theCartesianError)(3,1)=c31;
    (*theCartesianError)(3,2)=c32;
    (*theCartesianError)(3,3)=c33;
  }
  
  CartesianError3D(const AlgebraicSymMatrix & err) {
    if (err.num_row() == 3)
      theCartesianError = new AlgebraicSymMatrix(err);
    else {
      theCartesianError = 0;
      //throw DetLogicError("Not 3x3 Error Matrix: set pointer to 0");
      throw cms::Exception("DetLogicError")<<"Not 3x3 Error Matrix: set pointer to 0\n";
    }
  }

  CartesianError3D(const AlgebraicSymMatrix33 & err) :
    theCartesianError(new AlgebraicSymMatrix(asHepMatrix(err))) { }
 
  ~CartesianError3D() {}

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
  
  AlgebraicSymMatrix matrix() const {
    return *theCartesianError;
  }

  AlgebraicSymMatrix33 matrix_new() const {
    return asSMatrix<3>(*theCartesianError);
  }


  T rerr(const GlobalPoint& aPoint) const {
    T r2 = T(0.);
    if(aPoint.x() > T(0.) || aPoint.y() > T(0.))
      r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if(r2 > T(0.)) 
      return (1./r2)*(x2*cxx() + 2.*xy*cyx() + y2*cyy());
    else 
      return 0.5*(cxx() + cyy());  
  }

  T phierr(const GlobalPoint& aPoint) const {
    T r2 = T(0.);
    if(aPoint.x() > T(0.) || aPoint.y() > T(0.))
      r2 = aPoint.perp2();
    T x2 = aPoint.x()*aPoint.x();
    T y2 = aPoint.y()*aPoint.y();
    T xy = aPoint.x()*aPoint.y();
    if(r2 > T(0.)) 
      return (1./(r2*r2))*(y2*cxx() - 2.*xy*cyx() + x2*cyy());
    else
      return T(0.);
  }

  CartesianError3D operator+ (const CartesianError3D& err) const {
    return CartesianError3D( this->cxx()+err.cxx(), 
                           this->cyx()+err.cyx(),
                           this->cyy()+err.cyy(),
			   this->czx()+err.czx(),
			   this->czy()+err.czy(),
			   this->czz()+err.czz());
  }
  CartesianError3D operator- (const CartesianError3D& err) const {
    return CartesianError3D( this->cxx()-err.cxx(), 
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


