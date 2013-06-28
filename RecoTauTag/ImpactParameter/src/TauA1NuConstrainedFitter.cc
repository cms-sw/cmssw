/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TauA1NuConstrainedFitter.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"
#include <iostream>

TauA1NuConstrainedFitter::TauA1NuConstrainedFitter(unsigned int ambiguity,LorentzVectorParticle A1,TVector3 PVertex, TMatrixTSym<double> VertexCov):
  LagrangeMultipliersFitter(),
  MultiProngTauSolver(),
  ambiguity_(ambiguity)
{
  TLorentzVector Tau(0,0,0,0);
  LorentzVectorParticle Nu=EstimateNu(A1,PVertex, ambiguity_,Tau);
  particles_.push_back(A1);
  particles_.push_back(Nu);
  isconfigured=false;
  // setup 13 by 13 matrix
  int size=LorentzVectorParticle::NVertex+particles_.size()*LorentzVectorParticle::NLorentzandVertexPar;
  TMatrixT<double>    inpar(size,1);
  TMatrixTSym<double> incov(size);

  // Get primary vertex information
  if(VertexCov.GetNrows()!=LorentzVectorParticle::NVertex)return;
  inpar(LorentzVectorParticle::vx,0)=PVertex.X();
  inpar(LorentzVectorParticle::vy,0)=PVertex.Y();
  inpar(LorentzVectorParticle::vz,0)=PVertex.Z();  
  for(int i=0; i<LorentzVectorParticle::NVertex;i++){
    for(int j=0; j<LorentzVectorParticle::NVertex;j++)incov(i,j)=VertexCov(i,j);
  }
  int A1offset=LorentzVectorParticle::NVertex;
  int Nuoffset=LorentzVectorParticle::NLorentzandVertexPar+LorentzVectorParticle::NVertex;
  for(int i=0; i<LorentzVectorParticle::NLorentzandVertexPar;i++){
    inpar(i+A1offset,0)=A1.Parameter(i);
    inpar(i+Nuoffset,0)=Nu.Parameter(i)+1.0;// offset by 1 GeV to prevent convergence on first iteration
    for(int j=0; j<LorentzVectorParticle::NLorentzandVertexPar;j++){
      incov(i+A1offset,j+A1offset)=A1.Covariance(i,j);
      incov(i+Nuoffset,j+Nuoffset)=Nu.Covariance(i,j);
    }
  }
  // store expanded par for computation of final par (assumes fit has neglegible impact on a1 correlations with vertex errors)
  exppar.ResizeTo(nexpandedpar,1);
  exppar=ComputeInitalExpPar(inpar);
  expcov.ResizeTo(nexpandedpar,nexpandedpar);
  expcov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::ComputeInitalExpPar,inpar,incov);
  // store linearization point
  TMatrixT<double> PAR_0(npar,1);
  par_0.ResizeTo(npar);
  cov_0.ResizeTo(npar,npar);
  PAR_0=ComputeExpParToPar(exppar);
  for(int i=0; i<npar;i++)par_0(i)=PAR_0(i,0);
  cov_0=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::ComputeExpParToPar,exppar,expcov);
  for(int i=0; i<npar;i++){
    for(int j=0;j<npar;j++){cov_0(i,j)=expcov(i,j);}
  }
  // set up inital point for fit (cov handled in Fit() function)
  par.ResizeTo(npar);
  par=par_0;
  /*
  if(ambiguity_==zero){
    SolveAmbiguityAnalytically();
    isconfigured=true;
    isFit=true;
    return;
    }*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Check if Tau Direction is unphysical and if nessicary set the starting point to Theta_{GJ-Max} 
  TLorentzVector a1(par(a1_px),par(a1_py),par(a1_pz),sqrt(par(a1_m)*par(a1_m)+par(a1_px)*par(a1_px)+par(a1_py)*par(a1_py)+par(a1_pz)*par(a1_pz)));
  double phi(par(tau_phi)),theta(par(tau_theta));
  double scale=0.999;
  if(ambiguity==zero)scale=-1.0;
  if(SetTauDirectionatThetaGJMax(a1,theta,phi,scale)){
    TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
    TVector3 TauDir; TauDir.SetMagThetaPhi(1.0,theta,phi);
    SolvebyRotation(TauDir,a1,Tau_plus,Tau_minus,nu_plus,nu_minus);
    par(tau_phi)=phi;
    par(tau_theta)=theta;
    if(ambiguity_==plus){
      par(nu_px)=nu_plus.Px();
      par(nu_py)=nu_plus.Py();
      par(nu_pz)=nu_plus.Pz();
    }
    if(ambiguity_==minus){
      par(nu_px)=nu_minus.Px();
      par(nu_py)=nu_minus.Py();
      par(nu_pz)=nu_minus.Pz();
    }
    if(ambiguity_==zero){
      par(nu_px)=(nu_minus.Px()+nu_plus.Px())/2;
      par(nu_py)=(nu_minus.Py()+nu_plus.Py())/2;
      par(nu_pz)=(nu_minus.Pz()+nu_plus.Pz())/2;
    }
    if(ambiguity_==zero) par_0=par;
  }
  isconfigured=true;  
}


TMatrixT<double> TauA1NuConstrainedFitter::ComputeInitalExpPar(TMatrixT<double> &inpar){
  TMatrixT<double> outpar(nexpandedpar,1);
  int offset=LorentzVectorParticle::NVertex;// for A1
  TVector3 pv(inpar(LorentzVectorParticle::vx,0),inpar(LorentzVectorParticle::vy,0),inpar(LorentzVectorParticle::vz,0));
  TVector3 sv(inpar(LorentzVectorParticle::vx+offset,0),inpar(LorentzVectorParticle::vy+offset,0),inpar(LorentzVectorParticle::vz+offset,0));
  TVector3 TauDir=sv-pv;
  outpar(tau_phi,0)=TauDir.Phi();
  outpar(tau_theta,0)=TauDir.Theta();
  outpar(a1_px,0)=inpar(LorentzVectorParticle::px+offset,0);
  outpar(a1_py,0)=inpar(LorentzVectorParticle::py+offset,0);
  outpar(a1_pz,0)=inpar(LorentzVectorParticle::pz+offset,0);
  outpar(a1_m,0)=inpar(LorentzVectorParticle::m+offset,0);
  outpar(a1_vx,0)=inpar(LorentzVectorParticle::vx+offset,0);
  outpar(a1_vy,0)=inpar(LorentzVectorParticle::vy+offset,0);
  outpar(a1_vz,0)=inpar(LorentzVectorParticle::vz+offset,0);
  offset+=LorentzVectorParticle::NLorentzandVertexPar; // for Nu
  outpar(nu_px,0)=inpar(LorentzVectorParticle::px+offset,0);
  outpar(nu_py,0)=inpar(LorentzVectorParticle::py+offset,0);
  outpar(nu_pz,0)=inpar(LorentzVectorParticle::pz+offset,0);
  return outpar;
}


TMatrixT<double> TauA1NuConstrainedFitter::ComputeExpParToPar(TMatrixT<double> &inpar){
  TMatrixT<double> outpar(npar,1);
  for(int i=0;i<npar;i++){outpar(i,0)=inpar(i,0);}
  return outpar;
}


TMatrixT<double> TauA1NuConstrainedFitter::ComputeNuLorentzVectorPar(TMatrixT<double> &inpar){
  TMatrixT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar,1);
  outpar(LorentzVectorParticle::vx,0)=inpar(a1_vx,0);
  outpar(LorentzVectorParticle::vy,0)=inpar(a1_vy,0);
  outpar(LorentzVectorParticle::vz,0)=inpar(a1_vz,0);
  outpar(LorentzVectorParticle::px,0)=inpar(nu_px,0);
  outpar(LorentzVectorParticle::py,0)=inpar(nu_py,0);
  outpar(LorentzVectorParticle::pz,0)=inpar(nu_pz,0);
  outpar(LorentzVectorParticle::m,0)=0;
  return outpar;
}

TMatrixT<double> TauA1NuConstrainedFitter::ComputeA1LorentzVectorPar(TMatrixT<double> &inpar){
  TMatrixT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar,1);
  outpar(LorentzVectorParticle::vx,0)=inpar(a1_vx,0);
  outpar(LorentzVectorParticle::vy,0)=inpar(a1_vy,0);
  outpar(LorentzVectorParticle::vz,0)=inpar(a1_vz,0);
  outpar(LorentzVectorParticle::px,0)=inpar(a1_px,0);
  outpar(LorentzVectorParticle::py,0)=inpar(a1_py,0);
  outpar(LorentzVectorParticle::pz,0)=inpar(a1_pz,0);
  outpar(LorentzVectorParticle::m,0)=inpar(a1_m,0);
  return outpar;
}

TMatrixT<double> TauA1NuConstrainedFitter::ComputeMotherLorentzVectorPar(TMatrixT<double> &inpar){
  TMatrixT<double> outpar(LorentzVectorParticle::NLorentzandVertexPar,1);
  TMatrixT<double> nupar=ComputeNuLorentzVectorPar(inpar);
  TMatrixT<double> a1par=ComputeA1LorentzVectorPar(inpar);
  for(int i=0;i<LorentzVectorParticle::NLorentzandVertexPar;i++){
    if(i==LorentzVectorParticle::m)continue;
    if(i<LorentzVectorParticle::NVertex){outpar(i,0)=a1par(i,0);}
    else{outpar(i,0)=nupar(i,0)+a1par(i,0);}
  }
  double Enu2=pow(nupar(LorentzVectorParticle::px,0),2.0)+pow(nupar(LorentzVectorParticle::py,0),2.0)+pow(nupar(LorentzVectorParticle::pz,0),2.0);
  double Ea12=pow(a1par(LorentzVectorParticle::px,0),2.0)+pow(a1par(LorentzVectorParticle::py,0),2.0)+pow(a1par(LorentzVectorParticle::pz,0),2.0)+pow(a1par(LorentzVectorParticle::m,0),2.0);
  double P2=pow(outpar(LorentzVectorParticle::px,0),2.0)+pow(outpar(LorentzVectorParticle::py,0),2.0)+pow(outpar(LorentzVectorParticle::pz,0),2.0);
  outpar(LorentzVectorParticle::m,0)=sqrt(fabs(pow(sqrt(Enu2)+sqrt(Ea12),2.0)-P2));
  return outpar;
}

void TauA1NuConstrainedFitter::UpdateExpandedPar(){
  // assumes changes to a1 correlation to vertex is small
  if(par.GetNrows()==npar && cov.GetNrows() && exppar.GetNrows()==npar && expcov.GetNrows()) return;
  for(int i=0; i<npar;i++){
    exppar(i,0)=par(i);
    for(int j=0; j<npar;j++){expcov(i,j)=cov(i,j);}
  }
}

std::vector<LorentzVectorParticle> TauA1NuConstrainedFitter::GetReFitDaughters(){
  std::vector<LorentzVectorParticle> refitParticles;
  UpdateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_.at(i).Charge();b=particles_.at(i).BField();}
  TMatrixT<double> a1=ComputeA1LorentzVectorPar(exppar);
  TMatrixTSym<double> a1cov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::ComputeA1LorentzVectorPar,exppar,expcov);
  refitParticles.push_back(LorentzVectorParticle(a1,a1cov,fabs(PDGInfo::a_1_plus)*c,c,b));
  TMatrixT<double> nu=ComputeNuLorentzVectorPar(exppar);
  TMatrixTSym<double> nucov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::ComputeNuLorentzVectorPar,exppar,expcov);
  refitParticles.push_back(LorentzVectorParticle(nu,nucov,PDGInfo::nu_tau,0.0,b));
  return refitParticles;
}

LorentzVectorParticle TauA1NuConstrainedFitter::GetMother(){
  UpdateExpandedPar();
  double c(0),b(0);
  for(unsigned int i=0;i<particles_.size();i++){c+=particles_.at(i).Charge();b=particles_.at(i).BField();}
  TMatrixT<double> m=ComputeMotherLorentzVectorPar(exppar);
  TMatrixTSym<double> mcov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::ComputeMotherLorentzVectorPar,exppar,expcov);
  return LorentzVectorParticle(m,mcov,-1.0*fabs(PDGInfo::tau_minus)*c,c,b);
}

TVectorD TauA1NuConstrainedFitter::Value(TVectorD &v){
  TLorentzVector a1,nu;
  double phi(0),theta(0);
  TVector3 TauDir;
  CovertParToObjects(v,a1,nu,phi,theta,TauDir);
  TLorentzVector a1_d=a1;
  TLorentzVector nu_d=nu;
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  SolvebyRotation(TauDir,a1_d,Tau_plus,Tau_minus,nu_plus,nu_minus,false);
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


void TauA1NuConstrainedFitter::CovertParToObjects(TVectorD &v,TLorentzVector &a1,TLorentzVector &nu,double &phi,double &theta,TVector3 &TauDir){
  a1=TLorentzVector(v(a1_px),v(a1_py),v(a1_pz),sqrt(v(a1_m)*v(a1_m)+v(a1_px)*v(a1_px)+v(a1_py)*v(a1_py)+v(a1_pz)*v(a1_pz)));
  nu=TLorentzVector(v(nu_px),v(nu_py),v(nu_pz),sqrt(v(nu_px)*v(nu_px)+v(nu_py)*v(nu_py)+v(nu_pz)*v(nu_pz)));
  phi=v(tau_phi);
  theta=v(tau_theta);
  TauDir.SetMagThetaPhi(1.0,theta,phi);
}


bool TauA1NuConstrainedFitter::Fit(){
  ////////////////////////////////////////////
  // Run Kicker to force +/- solution to avoid solution being stuck in the local minimum
  if(ambiguity_==minus || ambiguity_==plus){ 
    TLorentzVector a1,nu;
    double phi(0),theta(0);
    TVector3 TauDir;
    CovertParToObjects(par,a1,nu,phi,theta,TauDir);
    TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus,nu_correct,nu_incorrect;
    if(ambiguity_==minus)SolvebyRotation(TauDir,a1,Tau_plus,Tau_minus,nu_incorrect,nu_correct,false);
    if(ambiguity_==plus)SolvebyRotation(TauDir,a1,Tau_plus,Tau_minus,nu_correct,nu_incorrect,false);
    nu.RotateZ(-phi);
    nu.RotateY(-theta);
    if(fabs(nu_incorrect.Pz()-nu.Pz())<fabs(nu_correct.Pz()-nu.Pz())){
      double pzkicked=nu_correct.Pz()-(nu_incorrect.Pz()-nu.Pz()); // minus sign is to make the kick a reflex about the ambiguity point 
      TLorentzVector nuKicked(nu.Px(),nu.Py(),pzkicked,sqrt(nu.Px()*nu.Px()+nu.Py()*nu.Py()+pzkicked*pzkicked));
      nuKicked.RotateY(-theta);
      nuKicked.RotateZ(-phi);
      par(nu_px)=nuKicked.Px();
      par(nu_py)=nuKicked.Py();
      par(nu_pz)=nuKicked.Pz();
    }
  }
  return LagrangeMultipliersFitter::Fit();
}


void TauA1NuConstrainedFitter::SolveAmbiguityAnalytically(){
  if(ambiguity_!=zero) return;
  std::cout << "TauA1NuConstrainedFitter::SolveAmbiguityAnalytically" << std::endl;
  TMatrixT<double> thepar=convertToMatrix(par_0);
  TMatrixT<double> angles=TauA1NuConstrainedFitter::FindThetaGJMax(thepar);
  TMatrixTSym<double> anglescov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::FindThetaGJMax,thepar,cov_0);
  std::cout << "TauA1NuConstrainedFitter::SolveAmbiguityAnalytically A" << std::endl;
  TMatrixT<double> thelpar(thepar.GetNrows()+2,1);
  TMatrixTSym<double> thelcov(thepar.GetNrows()+2);
  std::cout << "TauA1NuConstrainedFitter::SolveAmbiguityAnalytically B" << std::endl;
  for(int i=0;i<thepar.GetNrows();i++){
    thelpar(i,0)=thepar(i,0);
    for(int j=0;j<thepar.GetNrows();j++){thelcov(i,j)=cov_0(i,j);}
  }
  std::cout << "TauA1NuConstrainedFitter::SolveAmbiguityAnalytically C" << std::endl;
  thelpar(thelpar.GetNrows()-2,0)=anglescov(0,0);
  thelpar(thelpar.GetNrows()-1,0)=anglescov(1,1);
  par=convertToVector(TauA1NuConstrainedFitter::SetThetaGJMax(thelpar));
  std::cout << thelpar.GetNrows() << " " << par.GetNrows() << std::endl;
  cov.ResizeTo(par_0.GetNrows(),par_0.GetNrows());
  cov=ErrorMatrixPropagator::PropogateError(&TauA1NuConstrainedFitter::SetThetaGJMax,thelpar,thelcov);
  std::cout << "TauA1NuConstrainedFitter::SolveAmbiguityAnalytically done" << std::endl;
}

TMatrixT<double> TauA1NuConstrainedFitter::FindThetaGJMax(TMatrixT<double> &inpar){
  std::cout << "TauA1NuConstrainedFitter::FindThetaGJMax" << std::endl;
  TMatrixT<double> outpar(2,1);
  TVectorD thepar=convertToVector(inpar);
  TLorentzVector a1,nu;
  TVector3 TauDir;
  double phi,theta;
  CovertParToObjects(thepar,a1,nu,phi,theta,TauDir);
  outpar(0,0)=ThetaGJMax(a1);
  outpar(1,0)=TauDir.Dot(a1.Vect());
  std::cout << "TauA1NuConstrainedFitter::FindThetaGJMax done" << std::endl;
  return outpar;
}

TMatrixT<double> TauA1NuConstrainedFitter::SetThetaGJMax(TMatrixT<double> &inpar){
  std::cout << "TauA1NuConstrainedFitter::SetThetaGJMax" << std::endl;
  TMatrixT<double> outpar(inpar.GetNrows()-2,1);
  TVectorD thepar=convertToVector(inpar);
  TLorentzVector a1,nu;
  TVector3 TauDir;
  double phi,theta;
  double ErrthetaTau=inpar(inpar.GetNrows()-2,0);
  double ErrthetaA1=inpar(inpar.GetNrows()-1,0);
  CovertParToObjects(thepar,a1,nu,phi,theta,TauDir);
  TMatrixT<double> angles=TauA1NuConstrainedFitter::FindThetaGJMax(inpar);
  double delta=1;if(angles(1,0)!=0)delta=fabs(angles(0,0)/angles(1,0));
  double dtheta=(theta-a1.Theta());
  double dphi=fmod(fabs(phi-a1.Phi()),2*TMath::Pi());if(phi<a1.Phi())dphi*=-1.0;
  double scale=dtheta*ErrthetaTau/(ErrthetaTau+ErrthetaA1);
  outpar(tau_phi,0)=TauDir.Theta()+dtheta*delta*scale;
  outpar(tau_theta,0)=TauDir.Phi()+dphi*delta*scale;
  scale=dtheta*ErrthetaA1/(ErrthetaTau+ErrthetaA1);
  double a1theta=a1.Theta()-dtheta*delta*scale;
  double a1phi=a1.Phi()-dphi*delta*scale;
  a1.SetTheta(a1theta);
  a1.SetPhi(a1phi);
  outpar(a1_px,0)=a1.Px();
  outpar(a1_py,0)=a1.Py();
  outpar(a1_pz,0)=a1.Pz();
  TLorentzVector Tau_plus,Tau_minus,nu_plus,nu_minus;
  SolvebyRotation(TauDir,a1,Tau_plus,Tau_minus,nu_plus,nu_minus);
  outpar(nu_px,0)=nu_plus.Px();
  outpar(nu_py,0)=nu_plus.Py();
  outpar(nu_pz,0)=nu_plus.Pz();
  std::cout << "TauA1NuConstrainedFitter::SetThetaGJMax END" << std::endl;
  return outpar;
}

