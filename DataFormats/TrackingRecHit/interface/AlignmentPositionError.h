#ifndef ALIGNMENT_POSITION_ERROR_H
#define ALIGNMENT_POSITION_ERROR_H

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

/** The position error of a Det due to alignment.
 *  It is summed in quadrature with the RecHit global error.
 */

class AlignmentPositionError {

 public:

  AlignmentPositionError(){};
  
  AlignmentPositionError(float xx, float yy, float zz, float phixphix=0, float phiyphiy=0, float phizphiz=0);
 
  AlignmentPositionError(const GlobalErrorExtended& ge) : theGlobalError(ge) {};

  AlignmentPositionError(const GlobalError& ge);

  ~AlignmentPositionError(){};
  
  bool valid() const {
    return ( theGlobalError.cxx()>0 || theGlobalError.cyy()>0 || theGlobalError.czz()>0);
  }

  const GlobalErrorExtended & globalError() const { return theGlobalError; };

  AlignmentPositionError operator+ (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->globalError() + ape.globalError());
  };

  AlignmentPositionError operator- (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->globalError() - ape.globalError());

  };

  AlignmentPositionError & operator+= (const AlignmentPositionError& ape) {
    theGlobalError = GlobalErrorExtended(this->globalError() + ape.globalError());
    return *this;
  };

  AlignmentPositionError & operator-= (const AlignmentPositionError& ape) {
    theGlobalError = GlobalErrorExtended(this->globalError() - ape.globalError());
    return *this;
  };


 private:
  
  GlobalErrorExtended theGlobalError;
};

#endif // ALIGNMENT_POSITION_ERROR_H
