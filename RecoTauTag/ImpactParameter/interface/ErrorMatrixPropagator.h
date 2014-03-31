#ifndef ErrorMatrixPropagator_h
#define ErrorMatrixPropagator_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include <functional>

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVectorT.h"

namespace tauImpactParameter {

class  ErrorMatrixPropagator {
 public:
  ErrorMatrixPropagator(){};
  virtual ~ErrorMatrixPropagator(){};
  static TMatrixTSym<double> propagateError(std::function<TVectorT<double>(const TVectorT<double>&)> f, const TVectorT<double>& inPar, TMatrixTSym<double>& inCov, double epsilon=0.001, double errorEpsilonRatio=1000);
};

}
#endif


