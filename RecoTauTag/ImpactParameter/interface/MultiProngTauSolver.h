/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef MultiProngTauSolver_h
#define MultiProngTauSolver_h

#include "TVector3.h" 
#include "TLorentzVector.h" 
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"

class  MultiProngTauSolver {
 public:
  enum Ambiguity{zero,minus,plus,NAmbiguity};

  // constructor and Destructor
  MultiProngTauSolver(){};
  virtual ~MultiProngTauSolver(){};
      
  static void quadratic(double &x_plus,double &x_minus,double a, double b, double c);
  static void AnalyticESolver(TLorentzVector &nu_plus,TLorentzVector &nu_minus,TLorentzVector A1);
  static void NumericalESolver(TLorentzVector &nu_plus,TLorentzVector &nu_minus,TLorentzVector A1);
  static void SolvebyRotation(TVector3 TauDir,TLorentzVector A1, TLorentzVector &Tau_plus,TLorentzVector &Tau_minus,TLorentzVector &nu_plus,TLorentzVector &nu_minus,bool rotateback=true);
  static bool SetTauDirectionatThetaGJMax(TLorentzVector a1, double &theta,double &phi,double scale=1.0);  
  static double ThetaGJMax(TLorentzVector a1);
  static LorentzVectorParticle EstimateNu(LorentzVectorParticle &a1,TVector3 pv,int ambiguity,TLorentzVector &tau);

  static TMatrixT<double> RotateToTauFrame(TMatrixT<double> &inpar);
};

#endif
