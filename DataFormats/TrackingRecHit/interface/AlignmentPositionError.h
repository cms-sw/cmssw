#ifndef ALIGNMENT_POSITION_ERROR_H
#define ALIGNMENT_POSITION_ERROR_H

#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"

/** The position error of a Det due to alignment.
 *  It is summed in quadrature with the RecHit local error.
 */

class AlignmentPositionError {

 public:

  AlignmentPositionError(){};
  
  AlignmentPositionError(float xx, float yy, float phixphix, float phiyphiy); 
 
  AlignmentPositionError(const LocalErrorExtended& ge) : theLocalError(ge) {};

  ~AlignmentPositionError(){};
  
  //FIXME use all APEs for now
  bool valid() const {
    return true; //( theLocalError.cxx()>0 || theLocalError.cyy()>0 || theLocalError.cphixphix()>0 || theLocalError.cphiyphiy()>0);
  }

  const LocalErrorExtended & localError() const { return theLocalError; };

  AlignmentPositionError operator+ (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->localError() + ape.localError());
  };

  AlignmentPositionError operator- (const AlignmentPositionError& ape) const {
    return AlignmentPositionError ( this->localError() - ape.localError());

  };

  AlignmentPositionError & operator+= (const AlignmentPositionError& ape) {
    theLocalError = LocalErrorExtended(this->localError() + ape.localError());
    return *this;
  };

  AlignmentPositionError & operator-= (const AlignmentPositionError& ape) {
    theLocalError = LocalErrorExtended(this->localError() - ape.localError());
    return *this;
  };


 private:
  
  LocalErrorExtended theLocalError;
};

#endif // ALIGNMENT_POSITION_ERROR_H
