#ifndef ErrorMatrixPropagator_h
#define ErrorMatrixPropagator_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVectorT.h"

namespace tauImpactParameter {

class  ErrorMatrixPropagator {
 public:
  ErrorMatrixPropagator(){};
  virtual ~ErrorMatrixPropagator(){};
  static TMatrixTSym<double> propagateError(TVectorT<double> (*f)(const TVectorT<double> &par), const TVectorT<double>& inPar, TMatrixTSym<double>& inCov, double epsilon=0.001, double errorEpsilonRatio=1000);
};

}
#endif


