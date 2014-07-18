#ifndef RecoTauTag_ImpactParameter_TauA1NuConstrainedFitter_h
#define RecoTauTag_ImpactParameter_TauA1NuConstrainedFitter_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "TVector3.h"
#include "RecoTauTag/ImpactParameter/interface/MultiProngTauSolver.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include <vector>
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TMatrixTSym.h"

namespace tauImpactParameter {

class TauA1NuConstrainedFitter : public MultiProngTauSolver{
 public:
    TauA1NuConstrainedFitter(unsigned int ambiguity,const LorentzVectorParticle& A1,const TVector3& PVertex, const TMatrixTSym<double>& VertexCov);
  virtual ~TauA1NuConstrainedFitter(){};

  enum Pars{tau_phi=0,tau_theta,a1_px,a1_py,a1_pz,a1_m,nu_px,nu_py,nu_pz,npar};
  enum ExpandedPars{a1_vx=9,a1_vy,a1_vz,nexpandedpar};
  enum OrignialPars{norigpar=13};

  std::vector<LorentzVectorParticle> getRefitDaughters();
  LorentzVectorParticle getMother();
  double getTauRotationSignificance();

  bool fit();

 private:
  static TVectorT<double> ComputeInitalExpPar(const TVectorT<double> &inpar);
  static TVectorT<double> ComputeExpParToPar(const TVectorT<double> &inpar);
  static TVectorT<double> ComputeNuLorentzVectorPar(const TVectorT<double> &inpar);
  static TVectorT<double> ComputeA1LorentzVectorPar(const TVectorT<double> &inpar);
  static TVectorT<double> ComputeMotherLorentzVectorPar(const TVectorT<double> &inpar);
  static TVectorT<double> SolveAmbiguityAnalytically(const TVectorT<double> &inpar, unsigned int ambiguity);
  static TVectorT<double> SolveAmbiguityAnalyticallywithRot(const TVectorT<double> &inpar, unsigned int ambiguity);
  static TVectorT<double> TauRot(const TVectorT<double> &inpar);

  void UpdateExpandedPar();
  static void CovertParToObjects(const TVectorD &v,TLorentzVector &a1,TLorentzVector &nu,double &phi,double &theta,TVector3 &TauDir);

  TVectorT<double> par_0;
  TVectorT<double> par;
  TMatrixTSym<double> cov_0;
  TMatrixTSym<double> cov;

  TVectorT<double> exppar;
  TMatrixTSym<double> expcov;
  std::vector<LorentzVectorParticle> particles_;
  unsigned int ambiguity_;

  static  unsigned int static_amb;

};
}
#endif
