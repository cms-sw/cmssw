#ifndef LocalErrorType_H
#define LocalErrorType_H

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"

template <class T, class ErrorWeightType> 
class LocalErrorBaseExtended
{

public:
  /// Tag to request a null error matrix
  class NullMatrix{};

  LocalErrorBaseExtended() {}

  LocalErrorBaseExtended(const NullMatrix &) {}

   LocalErrorBaseExtended(InvalidError) {
    theCartesianError[0][0] = -9999.e10f;
    theCartesianError[0][1] = 0;
    theCartesianError[1][1] = -9999.e10f;
    theCartesianError[2][2] = -9999.e10f;
    theCartesianError[1][2] = 0;
    theCartesianError[1][3] = 0;
    theCartesianError[2][3] = 0;
    theCartesianError[3][3] = -9999.e10f;
  }

  bool invalid() const { return theCartesianError[0][0] <-1.e10f;}
  bool valid() const { return !invalid();}


  /**
   * Constructor.
   * The symmetric matrix stored as a lower triangular matrix
   */
  LocalErrorBaseExtended(T c11, T c21, T c31, T c41,
                 T c22, T c32, T c42,
                 T c33, T c43,
                 T c44) {
    theCartesianError(0,0)=c11;
    theCartesianError(1,0)=c21;
    theCartesianError(2,0)=c31;
    theCartesianError(3,0)=c41;

    theCartesianError(1,1)=c22;
    theCartesianError(2,1)=c32;
    theCartesianError(3,1)=c42;

    theCartesianError(2,2)=c33;
    theCartesianError(3,2)=c43;

    theCartesianError(3,3)=c44;

  }
 
    LocalErrorBaseExtended(const AlgebraicSymMatrix44 & err) : 
      theCartesianError(err) { }
  
  ~LocalErrorBaseExtended() {}

  T cxx() const {
    return theCartesianError(0,0);
  }
  
  T cyx() const {
    return theCartesianError(1,0);
  }
  
  T cphixx() const {
    return theCartesianError(2,0);
  }

  T cphiyx() const {
    return theCartesianError(3,0);
  }
 
  T cyy() const {
    return theCartesianError(1,1);
  }

  T cphixy() const {
    return theCartesianError(2,1);
  }

  T cphiyy() const {
    return theCartesianError(3,1);
  }

  T cphixphix() const {
    return theCartesianError(2,2);
  }

  T cphiyphix() const {
    return theCartesianError(3,2);
  }

  T cphiyphiy() const {
    return theCartesianError(3,3);
  }
 
 /**
   * Access method to the matrix,
   * /return The SymMatrix
   */
  const AlgebraicSymMatrix44 & matrix() const {
    return theCartesianError;
  }

  LocalErrorBaseExtended operator+ (const LocalErrorBaseExtended& err) const {
    return LocalErrorBaseExtended(theCartesianError + err.theCartesianError);
  }
  LocalErrorBaseExtended operator- (const LocalErrorBaseExtended& err) const {
    return LocalErrorBaseExtended(theCartesianError - err.theCartesianError);
  }
 
private:

  AlgebraicSymMatrix44 theCartesianError;

};

#endif
