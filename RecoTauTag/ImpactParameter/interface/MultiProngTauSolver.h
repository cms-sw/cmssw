#ifndef RecoTauTag_ImpactParameter_MultiProngTauSolver_h
#define RecoTauTag_ImpactParameter_MultiProngTauSolver_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "TVector3.h" 
#include "TLorentzVector.h" 
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"

namespace tauImpactParameter {

class  MultiProngTauSolver {
 public:
  enum Ambiguity{zero,minus,plus,NAmbiguity};

  // constructor and Destructor
  MultiProngTauSolver(){};
  virtual ~MultiProngTauSolver(){};
      
  static void quadratic(double& x_plus,double& x_minus,double a, double b, double c, bool &isReal);
  static void analyticESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus,const TLorentzVector& A1,bool &isReal);
  static void numericalESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus,const TLorentzVector& A1,bool &isReal);
  static void solveByRotation(const TVector3& TauDir,const TLorentzVector& A1, TLorentzVector& Tau_plus,TLorentzVector& Tau_minus,TLorentzVector& nu_plus,TLorentzVector& nu_minus, bool &isReal,bool rotateback=true);
  static bool setTauDirectionatThetaGJMax(const TLorentzVector& a1, double& theta,double& phi,double scale=1.0);  
  static double thetaGJMax(const TLorentzVector& a1);
  static LorentzVectorParticle estimateNu(const LorentzVectorParticle& a1, const TVector3& pv, int ambiguity, TLorentzVector& tau);
  static TVectorT<double> rotateToTauFrame(const TVectorT<double>& inpar);
};

}
#endif
