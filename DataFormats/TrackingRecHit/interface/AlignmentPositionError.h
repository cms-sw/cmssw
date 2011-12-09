#ifndef ALIGNMENT_POSITION_ERROR_H
#define ALIGNMENT_POSITION_ERROR_H

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

/** The position error of a Det due to alignment.
 *  It is summed in quadrature with the RecHit local error.
 */

class AlignmentPositionError {

 public:

  AlignmentPositionError(){};
  
  AlignmentPositionError(float dx, float dy, float dz);
  
  AlignmentPositionError(GlobalError ge) : theGlobalError(ge) {};

  ~AlignmentPositionError(){};
  

  bool valid() const {
    return ( theGlobalError.cxx()>0 || theGlobalError.cyy()>0 || theGlobalError.czz()>0 );
  }

  const GlobalError & globalError() const { return theGlobalError; };

  AlignmentPositionError operator+ (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->globalError() + ape.globalError());
  };

  AlignmentPositionError operator- (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->globalError() - ape.globalError());

  };

  AlignmentPositionError & operator+= (const AlignmentPositionError& ape) {
    theGlobalError = GlobalError(this->globalError() + ape.globalError());
    return *this;
  };

  AlignmentPositionError & operator-= (const AlignmentPositionError& ape) {
    theGlobalError = GlobalError(this->globalError() - ape.globalError());
    return *this;
  };


 private:
  
  GlobalError theGlobalError;
};

#endif // ALIGNMENT_POSITION_ERROR_H

