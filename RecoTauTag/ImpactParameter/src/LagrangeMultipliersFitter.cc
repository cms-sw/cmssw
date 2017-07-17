/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/ImpactParameter/interface/LagrangeMultipliersFitter.h"
#include "TDecompBK.h"
#include <iostream>

using namespace tauImpactParameter;

LagrangeMultipliersFitter::LagrangeMultipliersFitter():
  isConfigured_(false),
  isFit_(false),
  epsilon_(0.00001),
  weight_(1.0),
  maxDelta_(0.1),
  nitermax_(100),
  chi2_(1e10),
  D_(1,1),
  V_D_(1,1)
{}

bool LagrangeMultipliersFitter::fit(){
  if(cov_.GetNrows()!=par_0_.GetNrows()){
    // set cov to cov_0 until value is computed
    cov_.ResizeTo(par_0_.GetNrows(),par_0_.GetNrows());
    cov_=cov_0_;
  }
  if(!isConfigured_) return false;
  if(isFit_)return isConverged();
  isFit_=true;
  niter_=0;
  for(niter_=0;niter_<=nitermax_;niter_++){
    bool passed=applyLagrangianConstraints();
    if (!passed || (niter_==nitermax_ && delta_>=4.0*maxDelta_)) {
      edm::LogWarning("LagrangeMultipliersFitter::Fit") << "Reached Maximum number of iterations..." << niter_ << std::endl; 
      return false;
    }
    if(isConverged()) break;
  }
  computeVariance();
  return true;
}

bool LagrangeMultipliersFitter::applyLagrangianConstraints(){
  if(V_D_.GetNrows()!=nConstraints()) V_D_.ResizeTo(nConstraints(),nConstraints());
  if(D_.GetNrows()!=nConstraints() || D_.GetNcols()!=par_.GetNrows()) D_.ResizeTo(nConstraints(),par_.GetNrows());

  // Setup intial values
  TVectorT<double> alpha_A=par_;
  TVectorT<double> alpha_0=par_0_;
  TVectorT<double> delta_alpha_A=alpha_A-alpha_0;
  D_=derivative();
  TVectorT<double> d=value(par_);
  TVectorT<double> C=D_*delta_alpha_A-d;
  TMatrixTSym<double> V_alpha0=cov_0_;
  TMatrixTSym<double> V_D_inv=V_alpha0;
  V_D_inv.Similarity(D_);
  double det = V_D_inv.Determinant();
  TDecompBK Inverter(V_D_inv);
  if(fabs(det)>1e40){
    edm::LogWarning("LagrangeMultipliersFitter::Fit") << "Fit failed: unable to invert SYM gain matrix LARGE Determinant" << det << " \n" << std::endl;
    return false;
  }
  if(!Inverter.Decompose()){
    edm::LogWarning("LagrangeMultipliersFitter::Fit") << "Fit failed: unable to invert SYM gain matrix " << det << " \n" << std::endl;
    return false;
  }
  V_D_=Inverter.Invert();

  // solve equations
  TVectorT<double> lambda=-1.0*V_D_*C;
  TMatrixT<double> DT=D_; DT.T();
  TVectorT<double> alpha=alpha_0-V_alpha0*DT*lambda;

  // do while loop to see if the convergance criteria are satisfied
  double s(1), stepscale(0.01);
  chi2prev_=chi2_;
  double currentchi2(chiSquareUsingInitalPoint(alpha_A,lambda)), currentdelta(constraintDelta(par_));
  TVectorT<double> alpha_s=alpha;
  // convergence in 2 step procedure to minimize chi2 within MaxDelta_ of the constriants
  // 1) Get within 5x MaxDelta_
  // 2) converge based on improving chi2 and constrianed delta
  unsigned int Proc=ConstraintMin;
  if(constraintDelta(par_)<5*maxDelta_)Proc=Chi2AndConstaintMin;
  int  NIter=(int)(1.0/stepscale);
  for(int iter=0;iter<NIter;iter++){
    // compute safty cutoff for numberical constraint
    double diff=0;
    for(int l=0;l<alpha_s.GetNrows();l++){
      if(diff<alpha_s(l)-alpha_A(l))diff=alpha_s(l)-alpha_A(l);
    }
    double delta_alpha_s=constraintDelta(alpha_s);
    if(Proc==ConstraintMin){
      if(delta_alpha_s<currentdelta || iter==NIter || diff<100*epsilon_){currentchi2=chiSquareUsingInitalPoint(alpha_s,lambda); currentdelta=delta_alpha_s; ScaleFactor_=s; break;}
    }
    else if(Proc==Chi2AndConstaintMin){
      double chi2_s=chiSquareUsingInitalPoint(alpha_s,lambda);
      if((delta_alpha_s<currentdelta/*+maxDelta_*/ && chi2_s<currentchi2) || iter==NIter || diff<100*epsilon_){currentchi2=chi2_s; currentdelta=delta_alpha_s; ScaleFactor_=s; break;}
    }
    s-=stepscale;
    alpha_s=alpha_A+s*(alpha-alpha_A);
  }
  // set chi2
  chi2_=currentchi2;  
  //set delta
  delta_=currentdelta;
  par_=alpha_s;
  return true;
}

TMatrixD LagrangeMultipliersFitter::derivative(){ // alway evaluated at current par
  TMatrixD Derivatives(nConstraints(),par_.GetNrows());
  TVectorD par_plus(par_.GetNrows());
  TVectorD value_par(nConstraints());
  TVectorD value_par_plus(nConstraints());
  for(int j=0;j<par_.GetNrows();j++){
    for(int i=0;i<par_.GetNrows();i++){
      par_plus(i)=par_(i);
      if(i==j) par_plus(i)=par_(i)+epsilon_;
    }
    value_par=value(par_);
    value_par_plus=value(par_plus);
    for(int i=0; i<nConstraints();i++){
      Derivatives(i,j)=(value_par_plus(i)-value_par(i))/epsilon_;
    }
  }
  return Derivatives;
}

bool LagrangeMultipliersFitter::isConverged(){
  if(delta_<maxDelta_){
    return true;
  }
  return false;
}

double LagrangeMultipliersFitter::chiSquare(const TVectorT<double>& delta_alpha, const TVectorT<double>& lambda, const TMatrixT<double>& D, const TVectorT<double>& d){
  double c2=lambda*(D*delta_alpha+d);
  return c2;
}

double LagrangeMultipliersFitter::chiSquareUsingInitalPoint(const TVectorT<double>& alpha,const TVectorT<double>& lambda){
  if(cov_0_.GetNrows()!=V_alpha0_inv_.GetNrows()){
    TMatrixTSym<double> V_alpha0=cov_0_;
    V_alpha0_inv_.ResizeTo(cov_0_.GetNrows(),cov_0_.GetNrows());
    TDecompBK Inverter(V_alpha0);
    if(!Inverter.Decompose()){ // handle rare case where inversion is not possible (ie assume diagonal)
      edm::LogWarning("LagrangeMultipliersFitter::chiSquareUsingInitalPoint") << "Error non-invertable Matrix... Calculating under assumption that correlations can be neglected!!!" << std::endl;
      for(int j=0;j<par_.GetNrows();j++){
	for(int i=0;i<par_.GetNrows();i++){
	  if(i==j) V_alpha0_inv_(i,j)=1.0/V_alpha0(i,j);
	  else V_alpha0_inv_(i,j)=0.0;
	}
      }
    } else {
      V_alpha0_inv_=Inverter.Invert();
    }
  }

  TVectorT<double> alpha_0=par_0_;
  TVectorT<double> dalpha=alpha-alpha_0;
  double c2_var=dalpha*(V_alpha0_inv_*dalpha);
  TVectorT<double> alpha_v=alpha;
  double c2_constraints=lambda*value(alpha_v);
  double c2=c2_var+c2_constraints;
  return c2;
}

double LagrangeMultipliersFitter::constraintDelta(const TVectorT<double>& par){
  TVectorD d_par=value(par);
  double delta_d(0);
  for(int i = 0; i<d_par.GetNrows(); i++){
    delta_d+=fabs(d_par(i));
  }
  return delta_d;
}

TMatrixT<double> LagrangeMultipliersFitter::computeVariance(){
  TMatrixTSym<double> V_alpha0=cov_0_;
  TMatrixTSym<double> DTV_DD=V_D_.SimilarityT(D_);
  TMatrixT<double> DTV_DDV=DTV_DD*V_alpha0;
  TMatrixT<double> VDTV_DDV=V_alpha0*DTV_DDV;
  TMatrixT<double> CovCor=VDTV_DDV;
  //CovCor*=ScaleFactor_;
  if(V_corr_prev_.GetNrows()!=V_alpha0.GetNrows()){
    V_corr_prev_.ResizeTo(V_alpha0.GetNrows(),V_alpha0.GetNrows());
    V_corr_prev_=CovCor;
  }
  else{
    V_corr_prev_*=(1-ScaleFactor_);
    CovCor+=V_corr_prev_;
    V_corr_prev_=CovCor;
  }
  
  TMatrixT<double> V_alpha = V_alpha0-CovCor;
  for(int i=0; i<cov_.GetNrows();i++){
    for(int j=0; j<=i;j++){
      cov_(i,j)=V_alpha(i,j);
    }
  }
  return cov_;
}
