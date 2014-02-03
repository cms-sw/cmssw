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

TauA1NuConstrainedFitter::TauA1NuConstrainedFitter(unsigned int ambiguity,const LorentzVectorParticle& A1,const TVector3& PVertex, const TMatrixTSym<double>& VertexCov)
  : LagrangeMultipliersFitter(),
    MultiProngTauSolver(),
    ambiguity_(ambiguity)
{
  TLorentzVector Tau(0,0,0,0);
  LorentzVectorParticle Nu=estimateNu(A1,PVertex, ambiguity_,Tau);
  particles_.push_back(A1);
  particles_.push_back(Nu);
  isConfigured_=false;
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
  // store expanded par for computation of final par (assumes fit has neglegible impact on a1 correlations with vertex errors)
  exppar_.ResizeTo(nexpandedpar);
  exppar_=computeInitalExpPar(inpar);
  expcov_.ResizeTo(nexpandedpar,nexpandedpar);
  expcov_=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::computeInitalExpPar,inpar,incov);
  // store linearization point
  TVectorT<double> PAR_0(npar);
  par_0_.ResizeTo(npar);
  cov_0_.ResizeTo(npar,npar);
  PAR_0=computeExpParToPar(exppar_);
  for(int i=0; i<npar;i++)par_0_(i)=PAR_0(i);
  cov_0_=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::computeExpParToPar,exppar_,expcov_);
  for(int i=0; i<npar;i++){
    for(int j=0;j<npar;j++){cov_0_(i,j)=expcov_(i,j);}
  }
  // set up inital point for fit (cov handled in Fit() function)
  par_.ResizeTo(npar);
  par_=par_0_;
  /*
  if(ambiguity_==zero){
    aolveAmbiguityAnalytically();
    isConfigured_=true;
    isFit_=true;
    return;
    }*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Check if Tau Direction is unphysical and if nessicary set the starting point to Theta_{GJ-Max} 
  TLorentzVector a1(par_(a1_px),par_(a1_py),par_(a1_pz),sqrt(par_(a1_m)*par_(a1_m)+par_(a1_px)*par_(a1_px)+par_(a1_py)*par_(a1_py)+par_(a1_pz)*par_(a1_pz)));
  double phi(par_(tau_phi)),theta(par_(tau_theta));
  double scale=0.999;
  if(ambiguity==zero)scale=-1.0;
  if(setTauDirectionatThetaGJMax(a1,theta,phi,scale)){
    TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
    TVector3 TauDir; TauDir.SetMagThetaPhi(1.0,theta,phi);
    solveByRotation(TauDir,a1,Tau_plus,Tau_minus,nu_plus,nu_minus);
    par_(tau_phi)=phi;
    par_(tau_theta)=theta;
    if(ambiguity_==plus){
      par_(nu_px)=nu_plus.Px();
      par_(nu_py)=nu_plus.Py();
      par_(nu_pz)=nu_plus.Pz();
    }
    if(ambiguity_==minus){
      par_(nu_px)=nu_minus.Px();
      par_(nu_py)=nu_minus.Py();
      par_(nu_pz)=nu_minus.Pz();
    }
    if(ambiguity_==zero){
      par_(nu_px)=(nu_minus.Px()+nu_plus.Px())/2;
      par_(nu_py)=(nu_minus.Py()+nu_plus.Py())/2;
      par_(nu_pz)=(nu_minus.Pz()+nu_plus.Pz())/2;
    }
    if(ambiguity_==zero) par_0_=par_;
  }
  isConfigured_=true;  
}

TVectorT<double> TauA1NuConstrainedFitter::computeInitalExpPar(const TVectorT<double>& inpar){
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

TVectorT<double> TauA1NuConstrainedFitter::computeExpParToPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(npar,1);
  for(int i=0;i<npar;i++){outpar(i)=inpar(i);}
  return outpar;
}


TVectorT<double> TauA1NuConstrainedFitter::computeNuLorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar,1);
  outpar(LorentzVectorParticle::vx)=inpar(a1_vx);
  outpar(LorentzVectorParticle::vy)=inpar(a1_vy);
  outpar(LorentzVectorParticle::vz)=inpar(a1_vz);
  outpar(LorentzVectorParticle::px)=inpar(nu_px);
  outpar(LorentzVectorParticle::py)=inpar(nu_py);
  outpar(LorentzVectorParticle::pz)=inpar(nu_pz);
  outpar(LorentzVectorParticle::m)=0;
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::computeA1LorentzVectorPar(const TVectorT<double>& inpar){
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

TVectorT<double> TauA1NuConstrainedFitter::computeMotherLorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar);
  TVectorT<double> nupar=computeNuLorentzVectorPar(inpar);
  TVectorT<double> a1par=computeA1LorentzVectorPar(inpar);
  for(int i=0;i<LorentzVectorParticle::NLorentzandVertexPar;i++){
    if(i==LorentzVectorParticle::m)continue;
    if(i<LorentzVectorParticle::NVertex){outpar(i)=a1par(i);}
    else{outpar(i)=nupar(i)+a1par(i);}
  }
  double nu_px = nupar(LorentzVectorParticle::px);
  double nu_py = nupar(LorentzVectorParticle::py);
  double nu_pz = nupar(LorentzVectorParticle::pz); 
  double Enu2  = nu_px*nu_px + nu_py*nu_py + nu_pz*nu_pz;
  double a1_px = a1par(LorentzVectorParticle::px);
  double a1_py = a1par(LorentzVectorParticle::py);
  double a1_pz = a1par(LorentzVectorParticle::pz);
  double Ea12  = a1_px*a1_px + a1_py*a1_py + a1_pz*a1_pz;
  double outpar_px = outpar(LorentzVectorParticle::px);
  double outpar_py = outpar(LorentzVectorParticle::py);
  double outpar_pz = outpar(LorentzVectorParticle::pz);
  double P2=outpar_px*outpar_px + outpar_py*outpar_py + outpar_pz*outpar_pz;
  outpar(LorentzVectorParticle::m)=sqrt(fabs(Enu2 + Ea12 + 2*sqrt(Enu2*Ea12)-P2));
  return outpar;
}

void TauA1NuConstrainedFitter::updateExpandedPar(){
  // assumes changes to a1 correlation to vertex is small
  if(par_.GetNrows()==npar && cov_.GetNrows() && exppar_.GetNrows()==npar && expcov_.GetNrows()) return;
  for(int i=0; i<npar;i++){
    exppar_(i)=par_(i);
    for(int j=0; j<npar;j++){expcov_(i,j)=cov_(i,j);}
  }
}

std::vector<LorentzVectorParticle> TauA1NuConstrainedFitter::getRefitDaughters(){
  std::vector<LorentzVectorParticle> refitParticles;
  updateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_[i].charge();b=particles_[i].bField();}
  TVectorT<double> a1=computeA1LorentzVectorPar(exppar_);
  TMatrixTSym<double> a1cov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::computeA1LorentzVectorPar,exppar_,expcov_);
  refitParticles.push_back(LorentzVectorParticle(a1,a1cov,fabs(PDGInfo::a_1_plus)*c,c,b));
  TVectorT<double> nu=computeNuLorentzVectorPar(exppar_);
  TMatrixTSym<double> nucov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::computeNuLorentzVectorPar,exppar_,expcov_);
  refitParticles.push_back(LorentzVectorParticle(nu,nucov,PDGInfo::nu_tau,0.0,b));
  return refitParticles;
}

LorentzVectorParticle TauA1NuConstrainedFitter::getMother(){
  updateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_[i].charge();b=particles_[i].bField();}
  TVectorT<double> m=computeMotherLorentzVectorPar(exppar_);
  TMatrixTSym<double> mcov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::computeMotherLorentzVectorPar,exppar_,expcov_);
  return LorentzVectorParticle(m,mcov,-1.0*fabs(PDGInfo::tau_minus)*c,c,b);
}

TVectorD TauA1NuConstrainedFitter::value(const TVectorD& v){
  TLorentzVector a1,nu;
  double phi(0),theta(0);
  TVector3 TauDir;
  covertParToObjects(v,a1,nu,phi,theta,TauDir);
  TLorentzVector a1_d=a1;
  TLorentzVector nu_d=nu;
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  solveByRotation(TauDir,a1_d,Tau_plus,Tau_minus,nu_plus,nu_minus,false);
  a1.RotateZ(-phi);
  a1.RotateY(-theta);
  nu.RotateZ(-phi);
  nu.RotateY(-theta);
  TLorentzVector nufixed(-a1.Px(),-a1.Py(),nu.Pz(),sqrt(a1.Pt()*a1.Pt()+nu.Pz()*nu.Pz()));
  TLorentzVector tau=a1+nufixed;
  TVectorD d(3);
  if(ambiguity_==minus){    d(0)=sqrt(pow(nu.Pz()-nu_minus.Pz(),4.0)+pow(tau.M2()-PDGInfo::tau_mass()*PDGInfo::tau_mass(),2.0));}
  else if(ambiguity_==plus){d(0)=sqrt(pow(nu.Pz()-nu_plus.Pz(),4.0)+pow(tau.M2()-PDGInfo::tau_mass()*PDGInfo::tau_mass(),2.0));}
  else {d(0) = tau.M2()-PDGInfo::tau_mass()*PDGInfo::tau_mass();}
  d(1) = a1.Px()+nu.Px();
  d(2) = a1.Py()+nu.Py();
  return d;
}

void TauA1NuConstrainedFitter::covertParToObjects(const TVectorD& v, TLorentzVector& a1, TLorentzVector& nu, double& phi,double& theta, TVector3& TauDir){
  a1=TLorentzVector(v(a1_px),v(a1_py),v(a1_pz),sqrt(v(a1_m)*v(a1_m)+v(a1_px)*v(a1_px)+v(a1_py)*v(a1_py)+v(a1_pz)*v(a1_pz)));
  nu=TLorentzVector(v(nu_px),v(nu_py),v(nu_pz),sqrt(v(nu_px)*v(nu_px)+v(nu_py)*v(nu_py)+v(nu_pz)*v(nu_pz)));
  phi=v(tau_phi);
  theta=v(tau_theta);
  TauDir.SetMagThetaPhi(1.0,theta,phi);
}

bool TauA1NuConstrainedFitter::fit(){
  ////////////////////////////////////////////
  // Run Kicker to force +/- solution to avoid solution being stuck in the local minimum
  if(ambiguity_==minus || ambiguity_==plus){ 
    TLorentzVector a1,nu;
    double phi(0),theta(0);
    TVector3 TauDir;
    covertParToObjects(par_,a1,nu,phi,theta,TauDir);
    TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus,nu_correct,nu_incorrect;
    if(ambiguity_==minus) solveByRotation(TauDir,a1,Tau_plus,Tau_minus,nu_incorrect,nu_correct,false);
    if(ambiguity_==plus) solveByRotation(TauDir,a1,Tau_plus,Tau_minus,nu_correct,nu_incorrect,false);
    nu.RotateZ(-phi);
    nu.RotateY(-theta);
    if(fabs(nu_incorrect.Pz()-nu.Pz())<fabs(nu_correct.Pz()-nu.Pz())){
      double pzkicked=nu_correct.Pz()-(nu_incorrect.Pz()-nu.Pz()); // minus sign is to make the kick a reflex about the ambiguity point 
      TLorentzVector nuKicked(nu.Px(),nu.Py(),pzkicked,sqrt(nu.Px()*nu.Px()+nu.Py()*nu.Py()+pzkicked*pzkicked));
      nuKicked.RotateY(-theta);
      nuKicked.RotateZ(-phi);
      par_(nu_px)=nuKicked.Px();
      par_(nu_py)=nuKicked.Py();
      par_(nu_pz)=nuKicked.Pz();
    }
  }
  return LagrangeMultipliersFitter::fit();
}

void TauA1NuConstrainedFitter::solveAmbiguityAnalytically(){
  if(ambiguity_!=zero) return;
  TVectorT<double> angles=TauA1NuConstrainedFitter::findThetaGJMax(par_0_);
  TMatrixTSym<double> anglescov=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::findThetaGJMax,par_0_,cov_0_);
  TVectorT<double> thelpar(par_0_.GetNrows()+2,1);
  TMatrixTSym<double> thelcov(par_0_.GetNrows()+2);
  for(int i=0;i<par_0_.GetNrows();i++){
    thelpar(i)=par_0_(i);
    for(int j=0;j<par_0_.GetNrows();j++){thelcov(i,j)=cov_0_(i,j);}
  }
  thelpar(thelpar.GetNrows()-2)=anglescov(0,0);
  thelpar(thelpar.GetNrows()-1)=anglescov(1,1);
  par_=TauA1NuConstrainedFitter::setThetaGJMax(thelpar);
  cov_.ResizeTo(par_0_.GetNrows(),par_0_.GetNrows());
  cov_=ErrorMatrixPropagator::propagateError(&TauA1NuConstrainedFitter::setThetaGJMax,thelpar,thelcov);
}

TVectorT<double> TauA1NuConstrainedFitter::findThetaGJMax(const TVectorT<double>& inpar){
  TVectorT<double> outpar(2);
  TLorentzVector a1,nu;
  TVector3 TauDir;
  double phi,theta;
  covertParToObjects(inpar,a1,nu,phi,theta,TauDir);
  outpar(0)=thetaGJMax(a1);
  outpar(1)=TauDir.Dot(a1.Vect());
  return outpar;
}

TVectorT<double> TauA1NuConstrainedFitter::setThetaGJMax(const TVectorT<double>& inpar){
  TVectorT<double> outpar(inpar.GetNrows()-2);
  TLorentzVector a1,nu;
  TVector3 TauDir;
  double phi,theta;
  double ErrthetaTau=inpar(inpar.GetNrows()-2);
  double ErrthetaA1=inpar(inpar.GetNrows()-1);
  covertParToObjects(inpar,a1,nu,phi,theta,TauDir);
  TVectorT<double> angles=TauA1NuConstrainedFitter::findThetaGJMax(inpar);
  double delta=1;if(angles(1)!=0)delta=fabs(angles(0)/angles(1));
  double dtheta=(theta-a1.Theta());
  double dphi=fmod(fabs(phi-a1.Phi()),2*TMath::Pi());if(phi<a1.Phi())dphi*=-1.0;
  double scale=dtheta*ErrthetaTau/(ErrthetaTau+ErrthetaA1);
  outpar(tau_phi)=TauDir.Theta()+dtheta*delta*scale;
  outpar(tau_theta)=TauDir.Phi()+dphi*delta*scale;
  scale=dtheta*ErrthetaA1/(ErrthetaTau+ErrthetaA1);
  double a1theta=a1.Theta()-dtheta*delta*scale;
  double a1phi=a1.Phi()-dphi*delta*scale;
  a1.SetTheta(a1theta);
  a1.SetPhi(a1phi);
  outpar(a1_px)=a1.Px();
  outpar(a1_py)=a1.Py();
  outpar(a1_pz)=a1.Pz();
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  solveByRotation(TauDir,a1,Tau_plus,Tau_minus,nu_plus,nu_minus);
  outpar(nu_px)=nu_plus.Px();
  outpar(nu_py)=nu_plus.Py();
  outpar(nu_pz)=nu_plus.Pz();
  return outpar;
}

