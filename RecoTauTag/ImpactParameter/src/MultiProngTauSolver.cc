/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/MultiProngTauSolver.h"
#include <iostream>
#include "TMatrixTSym.h"

using namespace tauImpactParameter;

void MultiProngTauSolver::quadratic(double& x_plus,double& x_minus,double a, double b, double c){
  double R=b*b-4*a*c;
  if(R<0){R=0;}
  x_minus=(-b+sqrt(R))/(2.0*a); // opposite sign is smaller
  x_plus=(-b-sqrt(R))/(2.0*a);
}

void MultiProngTauSolver::analyticESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus, const TLorentzVector& A1){
  double a=(A1.Pz()*A1.Pz())/(A1.E()*A1.E())-1.0;
  double K=(PDGInfo::tau_mass()*PDGInfo::tau_mass()-A1.M2()-2.0*A1.Pt()*A1.Pt())/(2.0*A1.E());
  double b=2.0*K*A1.Pz()/A1.E();
  double c=K*K-A1.Pt()*A1.Pt();
  double z_plus(0),z_minus(0);
  quadratic(z_plus,z_minus,a,b,c);
  nu_plus.SetPxPyPzE(-A1.Px(),-A1.Py(),z_plus,sqrt(z_plus*z_plus+A1.Pt()*A1.Pt()));
  nu_minus.SetPxPyPzE(-A1.Px(),-A1.Py(),z_minus,sqrt(z_minus*z_minus+A1.Pt()*A1.Pt()));
}

void MultiProngTauSolver::numericalESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus, const TLorentzVector& A1){
  double rmin(-100), rmax(100), step(0.01), mtau2(PDGInfo::tau_mass()*PDGInfo::tau_mass()), z1(-999), z2(-999), zmin(-999), min(9999), prev(9999);
  double z=rmin;
  TLorentzVector nu,tau;
  for(int i=0;i<=(int)(rmax-rmin)/step;i++){
    nu.SetPxPyPzE(-A1.Px(),-A1.Py(),z,sqrt(z*z+A1.Pt()*A1.Pt()));
    tau=A1+nu;
    double m2=tau.M2();
    if(m2-mtau2<0 && prev-mtau2>=0) z1=z;
    if(m2-mtau2>0 && prev-mtau2<=0) z2=z;
    if(min>m2){ zmin=z; min=m2;}
    prev=m2;
    z+=step;
  }
  if(z1!=-999 && z2!=-999){
    nu_plus.SetPxPyPzE(-A1.Px(),-A1.Py(),z1,sqrt(z1*z1+A1.Pt()*A1.Pt()));
    nu_minus.SetPxPyPzE(-A1.Px(),-A1.Py(),z2,sqrt(z2*z2+A1.Pt()*A1.Pt()));
  }
  else{
    nu_plus.SetPxPyPzE(-A1.Px(),-A1.Py(),zmin,sqrt(zmin*zmin+A1.Pt()*A1.Pt()));
    nu_minus.SetPxPyPzE(-A1.Px(),-A1.Py(),zmin,sqrt(zmin*zmin+A1.Pt()*A1.Pt()));
  }
}

void MultiProngTauSolver::solveByRotation(const TVector3& TauDir,const TLorentzVector& A1, TLorentzVector& Tau_plus,TLorentzVector& Tau_minus,
					  TLorentzVector& nu_plus,TLorentzVector& nu_minus, bool rotateback){
  TLorentzVector A1rot=A1;
  double phi(TauDir.Phi()),theta(TauDir.Theta());
  A1rot.RotateZ(-phi);
  A1rot.RotateY(-theta);
  /////////////////////////////////////////////////////
  //  numericalESolver(nu_plus,nu_minus,A1rot); // for debugging AnalyticESolver (slow)
  //analyticESolver(nu_plus,nu_minus,A1rot);
  analyticESolver(nu_plus,nu_minus,A1rot);
  /////////////////////////////////////////////////////
  if(rotateback){
    nu_plus.RotateY(theta);
    nu_plus.RotateZ(phi);
    Tau_plus=A1+nu_plus;
    //
    nu_minus.RotateY(theta);
    nu_minus.RotateZ(phi);
    Tau_minus=A1+nu_minus;
  }
  else{
    Tau_plus=A1rot+nu_plus;
    Tau_minus=A1rot+nu_minus;
  }
}

bool MultiProngTauSolver::setTauDirectionatThetaGJMax(const TLorentzVector& a1, double& theta,double& phi,double scale){
  double thetaGJMax_a1 =thetaGJMax(a1);
  double dtheta=(theta-a1.Theta());
  double dphi=fmod(fabs(phi-a1.Phi()),2*TMath::Pi());if(phi<a1.Phi())dphi*=-1.0;
  double dphitheta=sqrt(dtheta*dtheta+dphi*dphi);
  if(thetaGJMax_a1<dphitheta || scale<0){
    theta=a1.Theta()+dtheta*(thetaGJMax_a1/dphitheta)*fabs(scale);
    phi=a1.Phi()+dphi*(thetaGJMax_a1/dphitheta)*fabs(scale);
    return true;
  }
  return false;
}

double MultiProngTauSolver::thetaGJMax(const TLorentzVector& a1){
  return asin(( PDGInfo::tau_mass()*PDGInfo::tau_mass()-a1.M2())/(2.0*PDGInfo::tau_mass()*fabs(a1.P())));
}

LorentzVectorParticle MultiProngTauSolver::estimateNu(const LorentzVectorParticle& a1, const TVector3& pv, int ambiguity, TLorentzVector& tau){
  TLorentzVector lorentzA1=a1.p4();
  TVector3 sv=a1.vertex();
  TVector3 tauFlghtDir=sv-pv;
  TLorentzVector nuGuess;

  TVector3 startingtauFlghtDir=tauFlghtDir.Unit();
  if(ambiguity==zero){
    double theta=tauFlghtDir.Theta();
    double phi=tauFlghtDir.Phi();
    setTauDirectionatThetaGJMax(lorentzA1,theta,phi);
    startingtauFlghtDir.SetMagThetaPhi(1.0,theta,phi);
  }
  TLorentzVector tau1,tau2,nu1,nu2;
  solveByRotation(startingtauFlghtDir,lorentzA1,tau1,tau2,nu1,nu2);
  if(ambiguity==plus){  nuGuess=nu1; tau=tau1; }
  if(ambiguity==minus){ nuGuess=nu2; tau=tau1; } 
  if(ambiguity==zero){  nuGuess=nu1; tau=tau1; }
  TVectorT<double> par(LorentzVectorParticle::NLorentzandVertexPar,10);
  par(LorentzVectorParticle::vx)=a1.parameter(LorentzVectorParticle::vx);
  par(LorentzVectorParticle::vy)=a1.parameter(LorentzVectorParticle::vy);
  par(LorentzVectorParticle::vz)=a1.parameter(LorentzVectorParticle::vz);
  par(LorentzVectorParticle::px)=nuGuess.Px(); 
  par(LorentzVectorParticle::py)=nuGuess.Py();
  par(LorentzVectorParticle::pz)=nuGuess.Pz();
  par(LorentzVectorParticle::m) =nuGuess.M();
  TMatrixTSym<double> Cov(LorentzVectorParticle::NLorentzandVertexPar);
  TMatrixTSym<double> pvCov=a1.vertexCov();
  for(int i=0; i<LorentzVectorParticle::NLorentzandVertexPar; i++){
    for(int j=0; j<=i; j++){
      if(i<LorentzVectorParticle::NVertex) Cov(i,j)=pvCov(i,j);
      else Cov(i,j)=0;
    }
    double v=0;
    if(i==LorentzVectorParticle::px || i==LorentzVectorParticle::py || i==LorentzVectorParticle::pz) v=10*par(i)*par(i);
    if(v<1000.0) v=1000.0; // try lowing to test impact
    Cov(i,i)+=v;
  }
  return LorentzVectorParticle(par,Cov,PDGInfo::nu_tau,0,a1.bField());
} 

TVectorT<double> MultiProngTauSolver::rotateToTauFrame(const TVectorT<double>& inpar){
  TVectorT<double> outpar(3);
  TVector3 res(inpar(0),inpar(1),inpar(2));
  TVector3 Uz;Uz.SetMagThetaPhi(1,inpar(4),inpar(3));
  res.RotateUz(Uz);
  outpar(0)=res.X();
  outpar(1)=res.Y();
  outpar(2)=res.Z();
  return outpar;
}
