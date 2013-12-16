/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/LagrangeMultipliersFitter.h"
#include "TDecompBK.h"
#include <iostream>

LagrangeMultipliersFitter::LagrangeMultipliersFitter():
  isconfigured(false),
  isFit(false),
  epsilon_(0.00001),
  weight_(1.0),
  MaxDelta_(0.1),
  nitermax_(100),
  chi2(1e10),
  D(1,1),
  V_D(1,1)
{

}

bool LagrangeMultipliersFitter::Fit(){
  if(cov.GetNrows()!=par_0.GetNrows()){
    // set cov to cov_0 until value is computed
    cov.ResizeTo(par_0.GetNrows(),par_0.GetNrows());
    cov=cov_0;
  }
  if(!isconfigured) return false;
  if(isFit)return isConverged();
  isFit=true;
  niter=0;
  for(niter=0;niter<=nitermax_;niter++){
    bool passed=ApplyLagrangianConstraints();
    if (!passed || (niter==nitermax_ && delta>=4.0*MaxDelta_)) {
      std::cout << "Reached Maximum number of iterations..." << niter << std::endl; 
      return false;
    }
    if(isConverged()) break;
  }
  ComputeVariance();
  return true;
}

bool LagrangeMultipliersFitter::ApplyLagrangianConstraints(){
  if(V_D.GetNrows()!=NConstraints()) V_D.ResizeTo(NConstraints(),NConstraints());
  if(D.GetNrows()!=NConstraints() || D.GetNcols()!=par.GetNrows()) D.ResizeTo(NConstraints(),par.GetNrows());

  // Setup intial values
  TMatrixT<double> alpha_A=convertToMatrix(par);
  TMatrixT<double> alpha_0=convertToMatrix(par_0);
  TMatrixT<double> delta_alpha_A=alpha_A-alpha_0;
  D=Derivative();
  TMatrixT<double> d=convertToMatrix(Value(par));
  TMatrixT<double> C=D*delta_alpha_A-d;
  TMatrixTSym<double> V_alpha0=cov_0;
  TMatrixTSym<double> V_D_inv=V_alpha0;
  V_D_inv.Similarity(D);
  double det = V_D_inv.Determinant();
  //std::cout << "LagrangeMultipliersFitter::ApplyLagrangianConstraints " << det << std::endl;
  TDecompBK Inverter(V_D_inv);
  if(fabs(det)>1e40){
    std::cout << "Fit failed: unable to invert SYM gain matrix LARGE Determinant" << det << " \n" << std::endl;
    return false;
  }
  if(!Inverter.Decompose()){
    std::cout << "Fit failed: unable to invert SYM gain matrix " << det << " \n" << std::endl;
    return false;
  }
  V_D=Inverter.Invert();

  // solve equations
  TMatrixT<double> lambda=-1.0*V_D*C;
  TMatrixT<double> DT=D; DT.T();
  TMatrixT<double> alpha=alpha_0-V_alpha0*DT*lambda;

  TVectorD finPar=convertToVector(alpha);

  // do while loop to see if the convergance criteria are satisfied
  double s(1), stepscale(0.01);
  chi2prev=chi2;
  double Curentchi2(ChiSquareUsingInitalPoint(alpha_A,lambda)), Currentdelta(ConstraintDelta(par));
  TMatrixT<double> alpha_s=alpha;
  // convergence in 2 step procedure to minimize chi2 within MaxDelta_ of the constriants
  // 1) Get within 5x MaxDelta_
  // 2) converge based on improving chi2 and constrianed delta
  unsigned int Proc=ConstraintMin;
  if(ConstraintDelta(par)<5*MaxDelta_)Proc=Chi2AndConstaintMin;
  int  NIter=(int)(1.0/stepscale);
  for(int iter=0;iter<NIter;iter++){
    // compute safty cutoff for numberical constraint
    double diff=0;
    for(int l=0;l<alpha_s.GetNrows();l++){
      if(diff<alpha_s(l,0)-alpha_A(l,0))diff=alpha_s(l,0)-alpha_A(l,0);
    }
    double delta_alpha_s=ConstraintDelta(convertToVector(alpha_s));
    if(Proc==ConstraintMin){
      if(delta_alpha_s<Currentdelta || iter==NIter || diff<100*epsilon_){Curentchi2=ChiSquareUsingInitalPoint(alpha_s,lambda); Currentdelta=delta_alpha_s; ScaleFactor=s; break;}
    }
    else if(Proc==Chi2AndConstaintMin){
      double chi2_s=ChiSquareUsingInitalPoint(alpha_s,lambda);
      if((delta_alpha_s<Currentdelta/*+MaxDelta_*/ && chi2_s<Curentchi2) || iter==NIter || diff<100*epsilon_){Curentchi2=chi2_s; Currentdelta=delta_alpha_s; ScaleFactor=s; break;}
    }
    s-=stepscale;
    alpha_s=alpha_A+s*(alpha-alpha_A);
  }
  // set chi2
  chi2=Curentchi2;  
  //set delta
  delta=Currentdelta;
  //std::cout << "LagrangeMultipliersFitter Chi^2 " << chi2 << " delta " << Currentdelta << std::endl; 
  //correct finPar to new stepsize
  par=convertToVector(alpha_s);
  return true;
}


TMatrixD LagrangeMultipliersFitter::Derivative(){ // alway evaluated at current par
  TMatrixD Derivatives(NConstraints(),par.GetNrows());
  TVectorD par_plus(par.GetNrows());
  TVectorD value(NConstraints());
  TVectorD value_plus(NConstraints());
  for(int j=0;j<par.GetNrows();j++){
    for(int i=0;i<par.GetNrows();i++){
      par_plus(i)=par(i);
      if(i==j) par_plus(i)=par(i)+epsilon_;
    }
    value=Value(par);
    value_plus=Value(par_plus);
    for(int i=0; i<NConstraints();i++){
      Derivatives(i,j)=(value_plus(i)-value(i))/epsilon_;
    }
  }
  return Derivatives;
}


bool LagrangeMultipliersFitter::isConverged(){
  if(delta<MaxDelta_ /*&& chi2prev-chi2<1.0 && chi2prev>chi2*/){
    //std::cout << "converged " << delta << " chi2 " <<  chi2 << " chi2prev " << chi2prev << std::endl; 
    return true;
  }
  return false;
}



TVectorT<double> LagrangeMultipliersFitter::convertToVector(TMatrixT<double> M){
  TVectorT<double> V(M.GetNrows()); 
  for(int i=0; i<M.GetNrows();i++){
    V(i)=M(i,0);
  }
  return V;
}


TMatrixT<double> LagrangeMultipliersFitter::convertToMatrix(TVectorT<double> V){
  TMatrixT<double> M(V.GetNrows(),1);
  for(int i=0; i<M.GetNrows();i++){
    M(i,0)=V(i);
  }
  return M;
}



double LagrangeMultipliersFitter::ChiSquare(TMatrixT<double> delta_alpha,TMatrixT<double> lambda,TMatrixT<double> D, TMatrixT<double> d){
  TMatrixT<double> lambdaT=lambda; lambdaT.T();
  TMatrixT<double> chisquare=lambdaT*(D*delta_alpha+d);
  double c2=chisquare(0,0);
  return c2;
}

double LagrangeMultipliersFitter::ChiSquareUsingInitalPoint(TMatrixT<double> alpha,TMatrixT<double> lambda){
  if(cov_0.GetNrows()!=V_alpha0_inv.GetNrows()){
    TMatrixTSym<double> V_alpha0=cov_0;
    V_alpha0_inv.ResizeTo(cov_0.GetNrows(),cov_0.GetNrows());
    TDecompBK Inverter(V_alpha0);
    if(!Inverter.Decompose()){ // handle rare case where inversion is not possible (ie assume diagonal)
      std::cout << "LagrangeMultipliersFitter::ChiSquareUsingInitalPoint: Error non-invertable Matrix... Calculating under assumption that correlations can be neglected!!!" << std::endl;
      for(int j=0;j<par.GetNrows();j++){
	for(int i=0;i<par.GetNrows();i++){
	  if(i==j) V_alpha0_inv(i,j)=1.0/V_alpha0(i,j);
	  else V_alpha0_inv(i,j)=0.0;
	}
      }
    }
    else{
      V_alpha0_inv=Inverter.Invert();
    }
  }

  TMatrixT<double> lambdaT=lambda; lambdaT.T();
  TMatrixT<double> alpha_0=convertToMatrix(par_0);
  TMatrixT<double> dalpha=alpha-alpha_0;
  TMatrixT<double> dalphaT=dalpha;  dalphaT.T();
  TMatrixT<double> chisquare_var=dalphaT*(V_alpha0_inv*dalpha);
  TVectorT<double> alpha_v=convertToVector(alpha);
  TMatrixT<double> chisquare_constraints=lambdaT*convertToMatrix(Value(alpha_v));
  double c2=chisquare_var(0,0)+chisquare_constraints(0,0);
  return c2;
}

double LagrangeMultipliersFitter::ConstraintDelta(TVectorT<double> par){
  TVectorD d_par=Value(par);
  double delta_d(0);
  for(int i = 0; i<d_par.GetNrows(); i++){
    delta_d+=fabs(d_par(i));
  }
  return delta_d;
}


TMatrixT<double>  LagrangeMultipliersFitter::ComputeVariance(){
  TMatrixTSym<double> V_alpha0=cov_0;
  TMatrixTSym<double> DTV_DD=V_D.SimilarityT(D);
  TMatrixT<double> DTV_DDV=DTV_DD*V_alpha0;
  TMatrixT<double> VDTV_DDV=V_alpha0*DTV_DDV;
  TMatrixT<double> CovCor=VDTV_DDV;
  //CovCor*=ScaleFactor;
  if(V_corr_prev.GetNrows()!=V_alpha0.GetNrows()){
    V_corr_prev.ResizeTo(V_alpha0.GetNrows(),V_alpha0.GetNrows());
    V_corr_prev=CovCor;
  }
  else{
    V_corr_prev*=(1-ScaleFactor);
    CovCor+=V_corr_prev;
    V_corr_prev=CovCor;
  }
  
  TMatrixT<double> V_alpha = V_alpha0-CovCor;
  for(int i=0; i<cov.GetNrows();i++){
    for(int j=0; j<=i;j++){
      cov(i,j)=V_alpha(i,j);
    }
  }
  return cov;
}
