#include "RecoTauTag/ImpactParameter/interface/MultiProngTauSolver.h"
#include "TMatrixTSym.h"
#include "TVectorT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace tauImpactParameter;

void MultiProngTauSolver::quadratic(double &x_plus,double &x_minus,double a, double b, double c, bool &isReal){
  double R=b*b-4*a*c;
  isReal=true;
  if(R<0){isReal=false;}// flag cases when R<0 but compute quadratic equation with |R|    
  x_minus=(-b+sqrt(fabs(R)))/(2.0*a); // opposite sign is smaller
  x_plus=(-b-sqrt(fabs(R)))/(2.0*a);
}

void MultiProngTauSolver::analyticESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus,const TLorentzVector &A1,bool &isReal){
  double a=(A1.Pz()*A1.Pz())/(A1.E()*A1.E())-1.0;
  double K=(PDGInfo::tau_mass()*PDGInfo::tau_mass()-A1.M2()-2.0*A1.Pt()*A1.Pt())/(2.0*A1.E());
  double b=2.0*K*A1.Pz()/A1.E();
  double c=K*K-A1.Pt()*A1.Pt();
  double z_plus(0),z_minus(0);
  quadratic(z_plus,z_minus,a,b,c,isReal);
  nu_plus=TLorentzVector(-A1.Px(),-A1.Py(),z_plus,sqrt(z_plus*z_plus+A1.Pt()*A1.Pt()));
  nu_minus=TLorentzVector(-A1.Px(),-A1.Py(),z_minus,sqrt(z_minus*z_minus+A1.Pt()*A1.Pt()));
}

void MultiProngTauSolver::numericalESolver(TLorentzVector& nu_plus,TLorentzVector& nu_minus,const TLorentzVector& A1,bool &isReal){
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
    nu_plus=TLorentzVector(-A1.Px(),-A1.Py(),z1,sqrt(z1*z1+A1.Pt()*A1.Pt()));
    nu_minus=TLorentzVector(-A1.Px(),-A1.Py(),z2,sqrt(z2*z2+A1.Pt()*A1.Pt()));
  }
  else{
    nu_plus=TLorentzVector(-A1.Px(),-A1.Py(),zmin,sqrt(zmin*zmin+A1.Pt()*A1.Pt()));
    nu_minus=TLorentzVector(-A1.Px(),-A1.Py(),zmin,sqrt(zmin*zmin+A1.Pt()*A1.Pt()));
  }
}

void MultiProngTauSolver::solveByRotation(const TVector3& TauDir,const TLorentzVector& A1, TLorentzVector& Tau_plus,TLorentzVector& Tau_minus,
					  TLorentzVector &nu_plus,TLorentzVector &nu_minus, bool &isReal,bool rotateback){
  TLorentzVector A1rot=A1;
  double phi(TauDir.Phi()),theta(TauDir.Theta());
  A1rot.RotateZ(-phi);
  A1rot.RotateY(-theta);
  /////////////////////////////////////////////////////
  //  NumericalESolver(nu_plus,nu_minus,A1rot); // for debugging analyticESolver (slow)
  analyticESolver(nu_plus,nu_minus,A1rot,isReal);
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
  double thetaGJMaxvar =thetaGJMax(a1);
  TVector3 a1v(a1.Vect()); if(a1v.Mag()!=0) a1v*=1/a1v.Mag();
  TVector3 tau(cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta));
  double dphitheta=acos(a1v.Dot(tau)/(a1v.Mag()*tau.Mag()));
  if(thetaGJMaxvar<dphitheta || scale<0){
    if(scale<0) scale=1.0;
    double a=(thetaGJMaxvar/dphitheta)-(1-scale);
    double b=1-(thetaGJMaxvar/dphitheta)+(1-scale);
    edm::LogInfo("RecoTauTag/ImpactParameter") << "SetTauDirectionatThetaGJMax before GF " <<  thetaGJMaxvar << " dot " << acos(a1v.Dot(tau)/(a1v.Mag()*tau.Mag())) << " a1 phi " <<  a1v.Phi() << " tau phi " << tau.Phi() << " a1 theta " <<a1v.Theta() << " tau theta " << tau.Theta()  ;
    tau*=a;
    a1v*=b;
    tau+=a1v;
    theta=tau.Theta();
    phi=tau.Phi();
    edm::LogInfo("RecoTauTag/ImpactParameter") << "SetTauDirectionatThetaGJMax GF " <<  thetaGJMaxvar << " dot " << acos(a1v.Dot(tau)/(a1v.Mag()*tau.Mag())) <<  " phi " << phi << " theta " << theta ;
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
    startingtauFlghtDir=TVector3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
  }
  TLorentzVector tau1,tau2,nu1,nu2;
  bool isReal;
  solveByRotation(startingtauFlghtDir,lorentzA1,tau1,tau2,nu1,nu2,isReal);
  if(ambiguity==plus){  nuGuess=nu1; tau=tau1; }
  if(ambiguity==minus){ nuGuess=nu2; tau=tau1; } 
  if(ambiguity==zero){  nuGuess=nu1; tau=tau1; }
  TVectorT<double>    par(LorentzVectorParticle::NLorentzandVertexPar);
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

TVectorT<double> MultiProngTauSolver::rotateToTauFrame(const TVectorT<double> &inpar){
  TVectorT<double> outpar(3);
  TVector3 res(inpar(0),inpar(1),inpar(2));
  TVector3 Uz(sin(inpar(4))*cos(inpar(3)),sin(inpar(4))*sin(inpar(3)),cos(inpar(4)));
  res.RotateUz(Uz);
  /*  double phi=inpar(3,0);
  double theta=inpar(4,0);
  res.RotateZ(-phi);
  TVector3 Y(0,1,0); 
  TVector3 thetadir=res.Cross(Y); 
  thetadir.RotateY(-theta);
  res.RotateY(-theta);
  res.RotateZ(thetadir.Phi());*/
  outpar(0)=res.X();
  outpar(1)=res.Y();
  outpar(2)=res.Z();
  return outpar;
}
