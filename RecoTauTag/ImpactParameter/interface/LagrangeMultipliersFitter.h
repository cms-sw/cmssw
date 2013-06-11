/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef LagrangeMultipliersFitter_H
#define LagrangeMultipliersFitter_H

#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TMatrixTSym.h"
#include <vector>

class LagrangeMultipliersFitter{
 public:
  enum Position{pos_x=0,pos_y,pos_z,nposdim};
  enum Parameters{par_vx=0,par_vy,par_vz,par_px,par_py,par_pz,par_m,npardim};
  enum ConvergeProc{ConstraintMin=0,Chi2Min,Chi2AndConstaintMin};

  LagrangeMultipliersFitter();
  virtual ~LagrangeMultipliersFitter(){};

  virtual void   SetWeight(double weight){weight_=weight;}
  virtual void   SetMaxDelta(double MaxDelta){MaxDelta_=MaxDelta;}
  virtual void   SetNIterMax(int Nitermax){nitermax_=Nitermax;}

  virtual bool Fit();
  virtual bool isConverged();
  virtual bool isConfigured(){return isconfigured;}
  virtual double ChiSquare(){return chi2;}
  virtual double CSum(){return delta;};
  virtual double NIter(){return niter;};
  virtual double NConstraints()=0;
  virtual double NDF()=0;
  virtual int    NDaughters()=0;

  virtual std::vector<LorentzVectorParticle> GetReFitDaughters()=0;
  virtual LorentzVectorParticle GetMother()=0;

  static TVectorT<double> convertToVector(TMatrixT<double> M);
  static TMatrixT<double> convertToMatrix(TVectorT<double> V);

 protected:
  virtual TVectorD Value(TVectorD &v)=0;

  TVectorD par_0; // parameter values for linearization point
  TVectorD par; // current parameter values
  TMatrixTSym<double> cov_0; //covariance matrix for linearization point (corresponding to par_0) 
  TMatrixTSym<double> cov; // current covariance matrix (corresponding to par) 
  bool isconfigured;
  bool isFit;

 private:
  bool  ApplyLagrangianConstraints();
  TMatrixT<double> Derivative();
  double ChiSquare(TMatrixT<double> delta_alpha,TMatrixT<double> lambda,TMatrixT<double> D,TMatrixT<double> d);
  double ChiSquareUsingInitalPoint(TMatrixT<double> alpha,TMatrixT<double> lambda);
  double ConstraintDelta(TVectorT<double> par);
  TMatrixT<double> ComputeVariance();

  // Configuration parameters
  double epsilon_,weight_,MaxDelta_,nitermax_;

  // Fit variables
  double chi2,chi2prev,delta,niter;

  // covariances and derivatives info
  TMatrixTSym<double> V_alpha0_inv;
  TMatrixT<double> D;
  TMatrixTSym<double> V_D;
  double ScaleFactor;
  TMatrixT<double> V_corr_prev;
  
};
#endif
