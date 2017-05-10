#ifndef RecoTauTag_ImpactParameter_LagrangeMultipliersFitter_h
#define RecoTauTag_ImpactParameter_LagrangeMultipliersFitter_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TMatrixTSym.h"
#include <vector>

namespace tauImpactParameter {

class LagrangeMultipliersFitter{
 public:
  enum Position{pos_x=0,pos_y,pos_z,nposdim};
  enum Parameters{par_vx=0,par_vy,par_vz,par_px,par_py,par_pz,par_m,npardim};
  enum ConvergeProc{ConstraintMin=0,Chi2Min,Chi2AndConstaintMin};

  LagrangeMultipliersFitter();
  virtual ~LagrangeMultipliersFitter(){};

  virtual void setWeight(double weight){weight_=weight;}
  virtual void setMaxDelta(double MaxDelta){maxDelta_=MaxDelta;}
  virtual void setNIterMax(int Nitermax){nitermax_=Nitermax;}

  virtual bool fit();
  virtual bool isConverged();
  virtual bool isConfigured(){return isConfigured_;}
  virtual double chiSquare(){return chi2_;}
  virtual double cSum(){return delta_;};
  virtual double nIter(){return niter_;};
  virtual double nConstraints()=0;
  virtual double ndf()=0;
  virtual int    nDaughters()=0;

  virtual std::vector<LorentzVectorParticle> getRefitDaughters()=0;
  virtual LorentzVectorParticle getMother()=0;

 protected:
  virtual TVectorD value(const TVectorD& v)=0;

  TVectorD par_0_; // parameter values for linearization point
  TVectorD par_; // current parameter values
  TMatrixTSym<double> cov_0_; //covariance matrix for linearization point (corresponding to par_0) 
  TMatrixTSym<double> cov_; // current covariance matrix (corresponding to par) 
  bool isConfigured_;
  bool isFit_;

 private:
  bool  applyLagrangianConstraints();
  TMatrixT<double> derivative();
  double chiSquare(const TVectorT<double>& delta_alpha, const TVectorT<double>& lambda, const TMatrixT<double>& D, const TVectorT<double>& d);
  double chiSquareUsingInitalPoint(const TVectorT<double>& alpha,const TVectorT<double>& lambda);
  double constraintDelta(const TVectorT<double>& par);
  TMatrixT<double> computeVariance();

  // Configuration parameters
  double epsilon_,weight_,maxDelta_,nitermax_;

  // Fit variables
  double chi2_,chi2prev_,delta_,niter_;

  // covariances and derivatives info
  TMatrixTSym<double> V_alpha0_inv_;
  TMatrixT<double> D_;
  TMatrixTSym<double> V_D_;
  double ScaleFactor_;
  TMatrixT<double> V_corr_prev_;
};

}
#endif
