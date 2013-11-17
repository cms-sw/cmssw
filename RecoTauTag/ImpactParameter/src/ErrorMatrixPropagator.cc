/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include "math.h"
#include <iostream>

TMatrixTSym<double> ErrorMatrixPropagator::PropogateError(TMatrixT<double> (*f)(TMatrixT<double> &par),TMatrixT<double> inPar,TMatrixTSym<double> inCov, double epsilon, double errorEpsilonRatio){
  TMatrixT<double> v=f(inPar);
  TMatrixT<double> Jacobian(inPar.GetNrows(),v.GetNrows());
  for(int i=0;i<inPar.GetNrows();i++){
    TMatrixT<double> ParPlusEpsilon=inPar;
    double error=sqrt(fabs(inCov(i,i)));
    double delta=epsilon;
    if(delta*errorEpsilonRatio<error) delta=error/errorEpsilonRatio;
    ParPlusEpsilon(i,0)+=delta;
    TMatrixT<double> vp=f(ParPlusEpsilon);
    for(int j=0;j<v.GetNrows();j++){Jacobian(i,j)=(vp(j,0)-v(j,0))/delta;}// Newtons approx.
  }
  TMatrixTSym<double> newCov=inCov.SimilarityT(Jacobian);
  /// debug
  /*
  std::cout << "ErrorMatrixPropagator::PropogateError inCov:" << std::endl;
  for(int i=0;i<inCov.GetNrows();i++){
    for(int j=0;j<inCov.GetNrows();j++){std::cout << inCov(i,j) << " ";}
    std::cout << std::endl;
  }
  std::cout << "ErrorMatrixPropagator::PropogateError Jacobian:" << std::endl;
  for(int i=0;i<Jacobian.GetNrows();i++){
    for(int j=0;j<Jacobian.GetNcols();j++){std::cout << Jacobian(i,j) << " ";}
    std::cout << std::endl;
  }
  TMatrixTSym<double> newCov=inCov.SimilarityT(Jacobian);
  std::cout << "ErrorMatrixPropagator::PropogateError newCov:" << std::endl;
  for(int i=0;i<newCov.GetNrows();i++){
    for(int j=0;j<newCov.GetNrows();j++){std::cout << newCov(i,j) << " ";}
    std::cout << std::endl;
    }*/
  return newCov;
}
