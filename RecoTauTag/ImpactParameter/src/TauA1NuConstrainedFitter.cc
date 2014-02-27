/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/ImpactParameter/interface/TauA1NuConstrainedFitter.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"
#include <iostream>

using namespace tauImpactParameter;

unsigned int TauA1NuConstrainedFitter::static_amb;

TauA1NuConstrainedFitter::TauA1NuConstrainedFitter(unsigned int ambiguity,const LorentzVectorParticle& A1,const TVector3& PVertex, const TMatrixTSym<double>& VertexCov):
  MultiProngTauSolver(),
  ambiguity_(ambiguity)
{
  TLorentzVector Tau(0,0,0,0);
  //dummy substitution not used later
  TVectorT<double> nu_par(LorentzVectorParticle::NLorentzandVertexPar,1);
  TMatrixTSym<double> nu_cov(LorentzVectorParticle::NLorentzandVertexPar); 
  LorentzVectorParticle Nu(nu_par,nu_cov,PDGInfo::nu_tau,0.0,A1.bField()); 
  particles_.push_back(A1);
  particles_.push_back(Nu);
  
  // setup 13 by 13 matrix
  int size=LorentzVectorParticle::NVertex+particles_.size()*LorentzVectorParticle::NLorentzandVertexPar;
  TVectorT<double>    inpar(size);
  TMatrixTSym<double> incov(size);

  // Get primary vertex information
  if(VertexCov.GetNrows()!=LorentzVectorParticle::NVertex)return;
  inpar(LorentzVectorParticle::vx)=PVertex.X();
  inpar(LorentzVectorParticle::vy)=PVertex.Y();
  inpar(LorentzVectorParticle::vz)=PVertex.Z();  
  for(int i=0; i<LorentzVectorParticle::NVertex;i++){
    for(int j=0; j<LorentzVectorParticle::NVertex;j++)incov(i,j)=VertexCov(i,j);
  }
  int A1offset=LorentzVectorParticle::NVertex;
  int Nuoffset=LorentzVectorParticle::NLorentzandVertexPar+LorentzVectorParticle::NVertex;
  for(int i=0; i<LorentzVectorParticle::NLorentzandVertexPar;i++){
    inpar(i+A1offset)=A1.parameter(i);
    inpar(i+Nuoffset)=Nu.parameter(i)+1.0;// offset by 1 GeV to prevent convergence on first iteration
    for(int j=0; j<LorentzVectorParticle::NLorentzandVertexPar;j++){
      incov(i+A1offset,j+A1offset)=A1.covariance(i,j);
      incov(i+Nuoffset,j+Nuoffset)=Nu.covariance(i,j);
    }
  }

  exppar.ResizeTo(nexpandedpar,1);
  exppar=ComputeInitalExpPar(inpar);
  expcov.ResizeTo(nexpandedpar,nexpandedpar);
  expcov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::ComputeInitalExpPar,inpar,incov);

  TVectorT<double> PAR_0(npar);
  par_0.ResizeTo(npar);
  cov_0.ResizeTo(npar,npar);
  PAR_0=ComputeExpParToPar(exppar);
  for(int i=0; i<npar;i++)par_0(i)=PAR_0(i);
  cov_0=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::ComputeExpParToPar,exppar,expcov);

  for(int i=0; i<npar;i++){
    for(int j=0;j<npar;j++){cov_0(i,j)=expcov(i,j);}
  }

  par.ResizeTo(npar);
  par=par_0;
  cov.ResizeTo(npar,npar);
  cov=cov_0;
}

TVectorT<double> TauA1NuConstrainedFitter::ComputeInitalExpPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(nexpandedpar);
  int offset=LorentzVectorParticle::NVertex;// for A1
  TVector3 pv(inpar(LorentzVectorParticle::vx),inpar(LorentzVectorParticle::vy),inpar(LorentzVectorParticle::vz));
  TVector3 sv(inpar(LorentzVectorParticle::vx+offset),inpar(LorentzVectorParticle::vy+offset),inpar(LorentzVectorParticle::vz+offset));
  TVector3 TauDir=sv-pv;
  outpar(tau_phi)=TauDir.Phi();
  outpar(tau_theta)=TauDir.Theta();
  outpar(a1_px)=inpar(LorentzVectorParticle::px+offset);
  outpar(a1_py)=inpar(LorentzVectorParticle::py+offset);
  outpar(a1_pz)=inpar(LorentzVectorParticle::pz+offset);
  outpar(a1_m)=inpar(LorentzVectorParticle::m+offset);
  outpar(a1_vx)=inpar(LorentzVectorParticle::vx+offset);
  outpar(a1_vy)=inpar(LorentzVectorParticle::vy+offset);
  outpar(a1_vz)=inpar(LorentzVectorParticle::vz+offset);
  offset+=LorentzVectorParticle::NLorentzandVertexPar; // for Nu
  outpar(nu_px)=inpar(LorentzVectorParticle::px+offset);
  outpar(nu_py)=inpar(LorentzVectorParticle::py+offset);
  outpar(nu_pz)=inpar(LorentzVectorParticle::pz+offset);
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::ComputeExpParToPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(npar);
  for(int i=0;i<npar;i++){outpar(i)=inpar(i);}
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::ComputeNuLorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar);
  outpar(LorentzVectorParticle::vx)=inpar(a1_vx);
  outpar(LorentzVectorParticle::vy)=inpar(a1_vy);
  outpar(LorentzVectorParticle::vz)=inpar(a1_vz);
  outpar(LorentzVectorParticle::px)=inpar(nu_px);
  outpar(LorentzVectorParticle::py)=inpar(nu_py);
  outpar(LorentzVectorParticle::pz)=inpar(nu_pz);
  outpar(LorentzVectorParticle::m)=0;
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::ComputeA1LorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar);
  outpar(LorentzVectorParticle::vx)=inpar(a1_vx);
  outpar(LorentzVectorParticle::vy)=inpar(a1_vy);
  outpar(LorentzVectorParticle::vz)=inpar(a1_vz);
  outpar(LorentzVectorParticle::px)=inpar(a1_px);
  outpar(LorentzVectorParticle::py)=inpar(a1_py);
  outpar(LorentzVectorParticle::pz)=inpar(a1_pz);
  outpar(LorentzVectorParticle::m)=inpar(a1_m);
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::ComputeMotherLorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar);
  TVectorT<double> nupar=ComputeNuLorentzVectorPar(inpar);
  TVectorT<double> a1par=ComputeA1LorentzVectorPar(inpar);
  for(int i=0;i<LorentzVectorParticle::NLorentzandVertexPar;i++){
    if(i<LorentzVectorParticle::NVertex){outpar(i)=a1par(i);}
    else{outpar(i)=nupar(i)+a1par(i);}
    //if(i==LorentzVectorParticle::m) outpar(i,0)=PDGInfo::tau_mass();
  }
  double nu_px = nupar(LorentzVectorParticle::px);
  double nu_py = nupar(LorentzVectorParticle::py);
  double nu_pz = nupar(LorentzVectorParticle::pz); 
  double Enu2  = nu_px*nu_px + nu_py*nu_py + nu_pz*nu_pz;
  double a1_px = a1par(LorentzVectorParticle::px);
  double a1_py = a1par(LorentzVectorParticle::py);
  double a1_pz = a1par(LorentzVectorParticle::pz);
  double a1_m =  a1par(LorentzVectorParticle::m);
  double Ea12  = a1_px*a1_px + a1_py*a1_py + a1_pz*a1_pz + a1_m*a1_m;
  double outpar_px = outpar(LorentzVectorParticle::px);
  double outpar_py = outpar(LorentzVectorParticle::py);
  double outpar_pz = outpar(LorentzVectorParticle::pz);
  double P2=outpar_px*outpar_px + outpar_py*outpar_py + outpar_pz*outpar_pz;
   outpar(LorentzVectorParticle::m)=sqrt(fabs(Enu2 + Ea12 + 2*sqrt(Enu2*Ea12)-P2));
  return outpar;
}

void TauA1NuConstrainedFitter::UpdateExpandedPar(){
  // assumes changes to a1 correlation to vertex is small
  if(par.GetNrows()==npar && cov.GetNrows()==npar && exppar.GetNrows()==npar && expcov.GetNrows()==npar) return;
  for(int i=0; i<npar;i++){
    exppar(i)=par(i);
    for(int j=0; j<npar;j++){expcov(i,j)=cov(i,j);}
  }
}

std::vector<LorentzVectorParticle> TauA1NuConstrainedFitter::getRefitDaughters(){
  std::vector<LorentzVectorParticle> refitParticles;
  UpdateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_[i].charge();b=particles_[i].bField();}
  TVectorT<double> a1=ComputeA1LorentzVectorPar(exppar);
  TMatrixTSym<double> a1cov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::ComputeA1LorentzVectorPar,exppar,expcov);
  refitParticles.push_back(LorentzVectorParticle(a1,a1cov,particles_[0].pdgId(),c,b));
  TVectorT<double> nu=ComputeNuLorentzVectorPar(exppar);
  TMatrixTSym<double> nucov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::ComputeNuLorentzVectorPar,exppar,expcov);
  refitParticles.push_back(LorentzVectorParticle(nu,nucov,PDGInfo::nu_tau,0.0,b));
  return refitParticles;
}

LorentzVectorParticle TauA1NuConstrainedFitter::getMother(){
  UpdateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_[i].charge();b=particles_[i].bField();}
  TVectorT<double> m=ComputeMotherLorentzVectorPar(exppar);
  TMatrixTSym<double> mcov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::ComputeMotherLorentzVectorPar,exppar,expcov);
  LorentzVectorParticle mymother= LorentzVectorParticle(m,mcov,(int)(-1.0*fabs(PDGInfo::tau_minus)*c),c,b);
  return mymother;
}

void TauA1NuConstrainedFitter::CovertParToObjects(const TVectorD &v,TLorentzVector &a1,TLorentzVector &nu,double &phi,double &theta,TVector3 &TauDir){
  a1=TLorentzVector(v(a1_px),v(a1_py),v(a1_pz),sqrt(v(a1_m)*v(a1_m)+v(a1_px)*v(a1_px)+v(a1_py)*v(a1_py)+v(a1_pz)*v(a1_pz)));
  nu=TLorentzVector(v(nu_px),v(nu_py),v(nu_pz),sqrt(v(nu_px)*v(nu_px)+v(nu_py)*v(nu_py)+v(nu_pz)*v(nu_pz)));
  phi=v(tau_phi);
  theta=v(tau_theta);
  TauDir.SetMagThetaPhi(1.0,theta,phi);
}

bool TauA1NuConstrainedFitter::fit(){
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Check if Tau Direction is unphysical and if nessicary set the starting point to Theta_{GJ-Max}
  TLorentzVector a1(par(a1_px),par(a1_py),par(a1_pz),sqrt(par(a1_m)*par(a1_m)+par(a1_px)*par(a1_px)+par(a1_py)*par(a1_py)+par(a1_pz)*par(a1_pz)));
  double phi(par(tau_phi)),theta(par(tau_theta));
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  TVector3 TauDir(cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta));
  bool isReal;
  solveByRotation(TauDir,a1,Tau_plus,Tau_minus,nu_plus,nu_minus,isReal);
  static_amb=ambiguity_;

  //check that the do product of the a1 and tau is positive, otherwise there is no information for tau direction -> use zero solution
  if(TauDir.Dot(a1.Vect())<0){
    isReal=false;
  }

  //case 1: is real then solve analytically
  if(isReal && (ambiguity_==plus || ambiguity_==minus)){
    // popogate errors
    TVectorT<double> par_tmp=TauA1NuConstrainedFitter::SolveAmbiguityAnalytically(par);
    cov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::SolveAmbiguityAnalytically,par,cov_0);
    for(int i=0; i<npar;i++) par(i)=par_tmp(i);
    return true;
  }
  // case 2 is in unphsyical region - rotate and substitue \theta_{GJ} with \theta_{GJ}^{Max} and then solve analytically
  else if(!isReal && ambiguity_==zero){
    TVectorT<double> par_tmp=TauA1NuConstrainedFitter::SolveAmbiguityAnalyticallywithRot(par);
    cov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::SolveAmbiguityAnalyticallywithRot,par,cov_0);
    for(int i=0; i<npar;i++) par(i)=par_tmp(i);
    return true;
  }
  return false;
}

TVectorT<double> TauA1NuConstrainedFitter::SolveAmbiguityAnalytically(const TVectorT<double>& inpar){
  // Solve equation quadratic equation
  TVectorT<double> outpar(inpar.GetNrows());
  TLorentzVector a1,nu;
  double phi(0),theta(0);
  TVector3 TauDir;
  CovertParToObjects(inpar,a1,nu,phi,theta,TauDir);
  TLorentzVector a1_d=a1;
  TLorentzVector nu_d=nu;
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  bool isReal;
  solveByRotation(TauDir,a1_d,Tau_plus,Tau_minus,nu_plus,nu_minus,isReal,true);
  if(static_amb==plus)nu=nu_plus;
  else nu=nu_minus;
  for(int i=0; i<outpar.GetNrows();i++){ outpar(i)=inpar(i);}
  outpar(nu_px)=nu.Px(); 
  outpar(nu_py)=nu.Py(); 
  outpar(nu_pz)=nu.Pz(); 
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::SolveAmbiguityAnalyticallywithRot(const TVectorT<double>& inpar){
  // Rotate and subsitute \theta_{GJ} with \theta_{GJ}^{Max} - assumes uncertianty on thata and phi of the a1 or small compared to the tau direction. 
  TVectorT<double> outpar(inpar.GetNrows());
  TLorentzVector a1,nu;
  double phi(0),theta(0);
  TVector3 TauDir;
  CovertParToObjects(inpar,a1,nu,phi,theta,TauDir);
  double theta_a1(a1.Theta()),phi_a1(a1.Phi()),theta_GJMax(thetaGJMax(a1));
  TauDir.RotateZ(-phi_a1);
  TauDir.RotateY(-theta_a1);
  double phiprime(TauDir.Phi());
  TauDir=TVector3(sin(theta_GJMax)*cos(phiprime),sin(theta_GJMax)*sin(phiprime),cos(theta_GJMax));
  TauDir.RotateY(theta_a1);
  TauDir.RotateZ(phi_a1);
  for(int i=0; i<outpar.GetNrows();i++) outpar(i)=inpar(i);
  outpar(tau_phi)=TauDir.Phi();
  outpar(tau_theta)=TauDir.Theta();
  return SolveAmbiguityAnalytically(outpar);
}

// Return the significance of the rotation when the tau direction is in the unphysical region
double TauA1NuConstrainedFitter::getTauRotationSignificance(){
  TVectorT<double> par_tmp=TauA1NuConstrainedFitter::TauRot(par);
  TMatrixTSym<double> cov_tmp=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::TauRot,par,cov_0);
  if(!(cov_tmp(0,0)>0)) return -999; // return invalid value if the covariance is unphysical (safety flag)
  if(par_tmp(0)>0)    return par_tmp(0)/sqrt(cov_tmp(0,0)); // return the significance if the value is in the unphysical region
  return 0; // reutrn 0 for the rotation significance if the tau is in the physical region
}


TVectorT<double> TauA1NuConstrainedFitter::TauRot(const TVectorT<double>& inpar){
  TVectorT<double> outpar(1);
  TLorentzVector a1,nu;
  double phi(0),theta(0);
  TVector3 TauDir;
  CovertParToObjects(inpar,a1,nu,phi,theta,TauDir);
  double theta_a1(a1.Theta()),phi_a1(a1.Phi()),theta_GJMax(thetaGJMax(a1));
  TauDir.RotateZ(-phi_a1);
  TauDir.RotateY(-theta_a1);
  outpar(0)=(TauDir.Theta()-theta_GJMax);
  return outpar;
}
