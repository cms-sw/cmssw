/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include "math.h"
#include <iostream>

using namespace tauImpactParameter;

TMatrixTSym<double> ErrorMatrixPropagator::propagateError(TVectorT<double> (*f)(const TVectorT<double>& par), 
							  const TVectorT<double>& inPar, TMatrixTSym<double>& inCov, double epsilon, double errorEpsilonRatio){
  TVectorT<double> v=f(inPar);
  TMatrixT<double> Jacobian(inPar.GetNrows(),v.GetNrows());
  for(int i=0;i<inPar.GetNrows();i++){
    TVectorT<double> ParPlusEpsilon=inPar;
    double error=sqrt(fabs(inCov(i,i)));
    double delta=epsilon;
    if(delta*errorEpsilonRatio<error) delta=error/errorEpsilonRatio;
    ParPlusEpsilon(i)+=delta;
    TVectorT<double> vp=f(ParPlusEpsilon);
    for(int j=0;j<v.GetNrows();j++){
      Jacobian(i,j)=(vp(j)-v(j))/delta;
    }// Newtons approx.
  }
  TMatrixTSym<double> newCov=inCov.SimilarityT(Jacobian);
  return newCov;
}
