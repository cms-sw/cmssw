#ifndef RecoTauTag_ImpactParameter_TauA1NuConstrainedFitter_h
#define RecoTauTag_ImpactParameter_TauA1NuConstrainedFitter_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "RecoTauTag/ImpactParameter/interface/LagrangeMultipliersFitter.h"
#include "TVector3.h"
#include "RecoTauTag/ImpactParameter/interface/MultiProngTauSolver.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include <vector>

namespace tauImpactParameter {

class TauA1NuConstrainedFitter : public LagrangeMultipliersFitter, public MultiProngTauSolver{
 public:
  TauA1NuConstrainedFitter(unsigned int ambiguity,const LorentzVectorParticle& A1,const TVector3& PVertex, const TMatrixTSym<double>& VertexCov);
  virtual ~TauA1NuConstrainedFitter(){};

  enum Pars{tau_phi=0,tau_theta,a1_px,a1_py,a1_pz,a1_m,nu_px,nu_py,nu_pz,npar};
  enum ExpandedPars{a1_vx=9,a1_vy,a1_vz,nexpandedpar};
  enum OrignialPars{norigpar=13};

  virtual bool fit();
  virtual double nConstraints(){return 3;}
  virtual double ndf(){return 0;}
  virtual int    nDaughters(){return 2;}

  std::vector<LorentzVectorParticle> getRefitDaughters();
  LorentzVectorParticle getMother();

 protected:
  virtual TVectorD value(const TVectorD& v);
    
 private:
  static TVectorT<double> computeInitalExpPar(const TVectorT<double>& inpar);
  static TVectorT<double> computeExpParToPar(const TVectorT<double>& inpar);
  static TVectorT<double> computeNuLorentzVectorPar(const TVectorT<double>& inpar);
  static TVectorT<double> computeA1LorentzVectorPar(const TVectorT<double>& inpar);
  static TVectorT<double> computeMotherLorentzVectorPar(const TVectorT<double>& inpar);
  void updateExpandedPar();
  static void covertParToObjects(const TVectorD& v, TLorentzVector& a1, TLorentzVector& nu, double& phi, double& theta, TVector3& TauDir);

  TVectorT<double> exppar_;
  TMatrixTSym<double> expcov_;
  std::vector<LorentzVectorParticle> particles_;
  int ConstraintMode_;
  unsigned int ambiguity_;

  void solveAmbiguityAnalytically();
  static TVectorT<double> findThetaGJMax(const TVectorT<double>& inpar);
  static TVectorT<double> setThetaGJMax(const TVectorT<double>& inpar);
};

}
#endif
