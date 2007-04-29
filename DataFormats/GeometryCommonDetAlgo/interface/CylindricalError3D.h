#ifndef _TRACKER_CYLINDRICAL_ERROR_3D_H_
#define _TRACKER_CYLINDRICAL_ERROR_3D_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/CartesianError3D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"

/** Obsolete.
 */

template <class T> class CylindricalError3D {

public:

  CylindricalError3D(): theCylindricalError(new AlgebraicSymMatrix(3,0)) {}

  CylindricalError3D(const CartesianError3D<T>& ge, const GlobalPoint& gp):
                        theCylindricalError(new AlgebraicSymMatrix(3,0)) {
    T r=gp.r();
    T x=gp.x();
    T y=gp.y();
    T cxx=ge.cxx();
    T cxy=ge.cyx();
    T cyy=ge.cyy();
    T czx=ge.czx();
    T czy=ge.czy();
    (*theCylindricalError)(1,1)=(pow(x,2)*cxx+2*x*y*cxy+pow(y,2)*cyy)/pow(r,2);
    (*theCylindricalError)(2,1)=((cyy-cxx)*x*y+cxy*pow(r,2))/pow(r,3);
    (*theCylindricalError)(2,2)=(pow(x,2)*cyy-2*x*y*cxy+pow(y,2)*cxx)/pow(r,4);
    (*theCylindricalError)(3,1)=(x*czx+y*czy)/r;
    (*theCylindricalError)(3,2)=(x*czy-y*czx)/pow(r,2);
    (*theCylindricalError)(3,3)=ge.czz();
  }

  CylindricalError3D(const AlgebraicSymMatrix & err) {
    if (err.num_row() == 3)
      theCylindricalError = new AlgebraicSymMatrix(err);
    else {
      theCylindricalError= 0;
      //throw DetLogicError("Not 3x3 Error Matrix: set pointer to 0");
      throw cms::Exception("DetLogicError")<<"Not 3x3 Error Matrix: set pointer to 0\n";
    }
  }

  CylindricalError3D(const AlgebraicSymMatrix33 & err) :
    theCylindricalError(new AlgebraicSymMatrix(asHepMatrix(err))) { }
 
  ~CylindricalError3D() {}

  T crr() const {
    return (*theCylindricalError)(1,1);
  }
  
  T crphi() const {
    return (*theCylindricalError)(2,1);
  }
  
  T cphiphi() const {
    return (*theCylindricalError)(2,2);
  }
  
  T czr() const {
    return (*theCylindricalError)(3,1);
  }
  
  T czphi() const {
    return (*theCylindricalError)(3,2);
  }
  
  T czz() const {
    return (*theCylindricalError)(3,3);
  }
  
  AlgebraicSymMatrix matrix() const {  
    return *theCylindricalError;
  }

  AlgebraicSymMatrix33 matrix_new() const {
    return asSMatrix<3>(*theCartesianError);
  }


private:

  DeepCopyPointer<AlgebraicSymMatrix> theCylindricalError;

};

#endif
