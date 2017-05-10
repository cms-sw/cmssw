/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "TDecompBK.h"
#include <iostream>

using namespace tauImpactParameter;

TrackHelixVertexFitter::TrackHelixVertexFitter(const std::vector<TrackParticle>& particles, const TVector3& vguess)
  : isFit_(false),
    isConfigured_(false),
    nParticles_(particles.size()),
    nPar_((NFreeTrackPar-NFreeVertexPar)*particles.size()+NFreeVertexPar),
    nVal_(TrackParticle::NHelixPar*particles.size())
{
  particles_=particles;
  par_.ResizeTo(nPar_);
  parcov_.ResizeTo(nPar_,nPar_);
  val_.ResizeTo(nVal_);
  cov_.ResizeTo(nVal_,nVal_);
  for(unsigned int p=0; p<particles.size();p++){
    for(unsigned int j=0; j<TrackParticle::NHelixPar;j++){
      val_(measuredValueIndex(j,p))=particles[p].parameter(j);
      for(unsigned int k=0; k<TrackParticle::NHelixPar;k++){
	cov_(measuredValueIndex(j,p),measuredValueIndex(k,p))=particles[p].covariance(j,k);
      }
    }
  }
  TDecompBK Inverter(cov_);
  double det = cov_.Determinant();
  if(!Inverter.Decompose()){
    edm::LogWarning("TrackHelixVertexFitter::TrackHelixVertexFitter") << "Fit failed: unable to invert SYM gain matrix " << det << " \n" << std::endl;
    return;
  }

  cov_inv_.ResizeTo(nVal_,nVal_);
  cov_inv_=Inverter.Invert();
  ndf_=nVal_-nPar_;
  // Set Initial conditions within reason
  par_(x0) = vguess.X(); parcov_(x0,x0)=1.0;
  par_(y0) = vguess.Y(); parcov_(y0,y0)=1.0;
  par_(z0) = vguess.Z(); parcov_(z0,z0)=1.0;
  for(unsigned int p=0; p<particles_.size();p++){
    par_(freeParIndex(kappa0,p))  = val_(measuredValueIndex(TrackParticle::kappa,p));
    par_(freeParIndex(lambda0,p)) = val_(measuredValueIndex(TrackParticle::lambda,p));
    par_(freeParIndex(phi0,p))    = val_(measuredValueIndex(TrackParticle::phi,p));

    parcov_(freeParIndex(kappa0,p),freeParIndex(kappa0,p))   = cov_(measuredValueIndex(TrackParticle::kappa,p),measuredValueIndex(TrackParticle::kappa,p));
    parcov_(freeParIndex(lambda0,p),freeParIndex(lambda0,p)) = cov_(measuredValueIndex(TrackParticle::lambda,p),measuredValueIndex(TrackParticle::lambda,p));
    parcov_(freeParIndex(phi0,p),freeParIndex(phi0,p))       = cov_(measuredValueIndex(TrackParticle::phi,p),measuredValueIndex(TrackParticle::phi,p));
  }
  isConfigured_=true;
}

TrackHelixVertexFitter::~TrackHelixVertexFitter(){}

double TrackHelixVertexFitter::updateChisquare(const TVectorT<double>& inpar){
  TVectorT<double> vprime=computePar(inpar);
  TVectorT<double> dalpha=vprime-val_;
  double c2=dalpha*(cov_inv_*dalpha);
  return c2;
}

std::vector<TrackParticle> TrackHelixVertexFitter::getRefitTracks(){
  std::vector<TrackParticle> refitParticles;
  for(unsigned int p=0;p<particles_.size();p++){
    TVectorT<double> FreePar(NFreeTrackPar);
    TMatrixTSym<double> FreeParCov(NFreeTrackPar);
    for(int i=0;i<FreeParCov.GetNrows();i++){
      FreePar(i)=par_(freeParIndex(i,p));
      for(int j=0;j<FreeParCov.GetNrows();j++){
	FreeParCov(i,j)=parcov_(freeParIndex(i,p),freeParIndex(j,p));
      }
    }
    TVectorT<double>    TrackPar=computePar(FreePar);
    TMatrixTSym<double> TrackCov=ErrorMatrixPropagator::propagateError(&TrackHelixVertexFitter::computePar,FreePar,FreeParCov);
    refitParticles.push_back(TrackParticle(TrackPar,TrackCov,particles_[p].pdgId(),particles_[p].mass(),particles_[p].charge(),particles_[p].bField()));
  }
  return refitParticles;
}

std::vector<LorentzVectorParticle> TrackHelixVertexFitter::getRefitLorentzVectorParticles(){
  std::vector<LorentzVectorParticle> refitParticles;
  for(unsigned int p=0;p<particles_.size();p++){
    TVectorT<double>    FreePar(NFreeTrackPar+NExtraPar+MassOffSet);
    TMatrixTSym<double> FreeParCov(NFreeTrackPar+NExtraPar+MassOffSet);
    for(int i=0;i<NFreeTrackPar;i++){
      FreePar(i)=par_(freeParIndex(i,p));
      for(int j=0;j<NFreeTrackPar;j++){
        FreeParCov(i,j)=parcov_(freeParIndex(i,p),freeParIndex(j,p));
      }
    }
    FreePar(NFreeTrackPar+MassOffSet)=particles_[p].mass();
    FreePar(NFreeTrackPar+BField0)=particles_[p].bField();
    TVectorT<double>    LVPar=computeLorentzVectorPar(FreePar);
    TMatrixTSym<double> LVCov=ErrorMatrixPropagator::propagateError(&TrackHelixVertexFitter::computeLorentzVectorPar,FreePar,FreeParCov);
    refitParticles.push_back(LorentzVectorParticle(LVPar,LVCov,particles_[p].pdgId(),particles_[p].charge(),particles_[p].bField()));
  }
  return refitParticles;
}

LorentzVectorParticle TrackHelixVertexFitter::getMother(int pdgid){
  double c(0),b(0);
  TVectorT<double>    FreePar(par_.GetNrows()+NExtraPar+particles_.size());
  TMatrixTSym<double> FreeParCov(par_.GetNrows()+NExtraPar+particles_.size());
  for(int i=0;i<par_.GetNrows();i++){
    FreePar(i)=par_(i);
    for(int j=0;j<par_.GetNrows();j++){FreeParCov(i,j)=parcov_(i,j);}
  }
  for(unsigned int p=0; p<particles_.size();p++){
    b=particles_[p].bField();
    c+=particles_[p].charge();
    FreePar(par_.GetNrows()+MassOffSet+p)=particles_[p].mass();
  }
  FreePar(par_.GetNrows()+BField0)=b;
  TVectorT<double>    mpar=computeMotherLorentzVectorPar(FreePar);
  TMatrixTSym<double> mcov=ErrorMatrixPropagator::propagateError(&TrackHelixVertexFitter::computeMotherLorentzVectorPar,FreePar,FreeParCov);
  return LorentzVectorParticle(mpar,mcov,pdgid,c,b);
}

TVector3 TrackHelixVertexFitter::getVertex(){
  return TVector3(par_(freeParIndex(x0,0)),par_(freeParIndex(y0,0)),par_(freeParIndex(z0,0)));
}

TMatrixTSym<double> TrackHelixVertexFitter::getVertexError(){
  TMatrixTSym<double> c(NFreeVertexPar);
  for(unsigned int i=0;i<NFreeVertexPar;i++){
    for(unsigned int j=0;j<NFreeVertexPar;j++){c(freeParIndex(i,0),freeParIndex(j,0))=parcov_(freeParIndex(i,0),freeParIndex(j,0));}
  }
  return c;
}

void TrackHelixVertexFitter::computedxydz(const TVectorT<double>& inpar,int p,double& kappa,double& lam,double& phi,double& x,double& y,double& z,double& s,double& dxy,double& dz){
  kappa=inpar(freeParIndex(kappa0,p));
  lam=inpar(freeParIndex(lambda0,p));
  phi=inpar(freeParIndex(phi0,p));
  x=inpar(freeParIndex(x0,p));
  y=inpar(freeParIndex(y0,p));
  z=inpar(freeParIndex(z0,p));
  double v=(2.0*kappa*(x*cos(phi)+y*sin(phi)));
  double arcsinv=0;
  if(v>=1.0){arcsinv=TMath::Pi()/2;}
  else if(v<=-1.0){arcsinv=-TMath::Pi()/2;}
  else{arcsinv=asin(v);}
  s=1.0/(2.0*kappa)*arcsinv;
  dxy=y*cos(phi)-x*sin(phi)-(1/kappa)*sin(kappa*s)*sin(kappa*s);
  dz=z-s*tan(lam);
}

TVectorT<double> TrackHelixVertexFitter::computePar(const TVectorT<double>& inpar){
  int nparticles=(inpar.GetNrows()-NFreeVertexPar)/(NFreeTrackPar-NFreeVertexPar);
  TVectorT<double> helices(nparticles*TrackParticle::NHelixPar);
  for(int p=0;p<nparticles;p++){
    TVectorT<double> TrackPar=computeTrackPar(inpar,p);
    for(int i=0;i<TrackParticle::NHelixPar;i++){helices(measuredValueIndex(i,p))=TrackPar(i);}
  }
  return helices;
}

TVectorT<double> TrackHelixVertexFitter::computeTrackPar(const TVectorT<double>& inpar, int p){
  TVectorT<double> helix(TrackParticle::NHelixPar);
  // copy parameters that are 1 to 1
  double kappa,lam,phi,x,y,z,s,dxy,dz;
  TrackHelixVertexFitter::computedxydz(inpar,p,kappa,lam,phi,x,y,z,s,dxy,dz);
  helix(TrackParticle::kappa)  = kappa;
  helix(TrackParticle::lambda) = lam;
  helix(TrackParticle::phi)    = phi;
  helix(TrackParticle::dxy)    = dxy;
  helix(TrackParticle::dz)     = dz;
  return helix;
}

TVectorT<double> TrackHelixVertexFitter::computeLorentzVectorPar(const TVectorT<double>& inpar){
  int np(0), parsize(0); parSizeInfo(inpar,np,parsize,true);
  double B=inpar(parsize+BField0);
  double massHypothesis=inpar(parsize+MassOffSet);
  TVectorT<double> LV(LorentzVectorParticle::NLorentzandVertexPar);
  double kappa,lam,phi,x,y,z,s,dxy,dz;
  int p=0;
  TrackHelixVertexFitter::computedxydz(inpar,p,kappa,lam,phi,x,y,z,s,dxy,dz);
  double phi1 = 2*s*kappa+phi;
  double bOverK = B*(1.0/fabs(kappa));
  LV(LorentzVectorParticle::px) = bOverK*cos(phi1);
  LV(LorentzVectorParticle::py) = bOverK*sin(phi1);
  LV(LorentzVectorParticle::pz) = bOverK*tan(lam) ;
  LV(LorentzVectorParticle::m)  = massHypothesis;
  LV(LorentzVectorParticle::vx) = x;
  LV(LorentzVectorParticle::vy) = y;
  LV(LorentzVectorParticle::vz) = z;
  return LV;
}

TVectorT<double> TrackHelixVertexFitter::computeMotherLorentzVectorPar(const TVectorT<double>& inpar){
  TVectorT<double> mother(LorentzVectorParticle::NLorentzandVertexPar);
  double E(0);
  int np(0), parsize(0); parSizeInfo(inpar,np,parsize,true);
  for(int p=0;p<np;p++){
    TVectorT<double> particlepar(NFreeTrackPar+NExtraPar+MassOffSet);
    for(int i=0;i<NFreeTrackPar;i++){particlepar(i)=inpar(freeParIndex(i,p));}
    particlepar(NFreeTrackPar+BField0)=inpar(parsize+BField0);
    particlepar(NFreeTrackPar+MassOffSet)=inpar(parsize+MassOffSet+p);
    TVectorT<double> daughter=TrackHelixVertexFitter::computeLorentzVectorPar(particlepar);
    mother(LorentzVectorParticle::px)+=daughter(LorentzVectorParticle::px);
    mother(LorentzVectorParticle::py)+=daughter(LorentzVectorParticle::py);
    mother(LorentzVectorParticle::pz)+=daughter(LorentzVectorParticle::pz);
    mother(LorentzVectorParticle::vx)=daughter(LorentzVectorParticle::vx);
    mother(LorentzVectorParticle::vy)=daughter(LorentzVectorParticle::vy);
    mother(LorentzVectorParticle::vz)=daughter(LorentzVectorParticle::vz);
    E+=sqrt((daughter(LorentzVectorParticle::px)*daughter(LorentzVectorParticle::px)+
	     daughter(LorentzVectorParticle::py)*daughter(LorentzVectorParticle::py)+
	     daughter(LorentzVectorParticle::pz)*daughter(LorentzVectorParticle::pz)+
	     daughter(LorentzVectorParticle::m)*daughter(LorentzVectorParticle::m)));
  }
  double P2=(mother(LorentzVectorParticle::px)*mother(LorentzVectorParticle::px)+
	     mother(LorentzVectorParticle::py)*mother(LorentzVectorParticle::py)+
	     mother(LorentzVectorParticle::pz)*mother(LorentzVectorParticle::pz));
  mother(LorentzVectorParticle::m)=(E*E-P2)/sqrt(fabs(E*E-P2));
  return mother;
}

TString TrackHelixVertexFitter::freeParName(int Par){
  int p(0);
  if(Par==x0)     return "x0";
  if(Par==y0)     return "y0";
  if(Par==z0)     return "z0";
  for(p=0;p<nParticles_;p++){
    if((Par-NFreeVertexPar)<(p+1)*(NFreeTrackPar-NFreeVertexPar))break;
  }
  TString n;
  int index=Par-p*(NFreeTrackPar-NFreeVertexPar);
  if(index==kappa0)  n="kappa0";
  if(index==lambda0) n="lambda0";
  if(index==phi0)    n="phi0";
  n+="_particle";n+=p;
  return n;
}

void TrackHelixVertexFitter::parSizeInfo(const TVectorT<double>& inpar, int& np, int& parsize, bool hasextras){
  if(hasextras)np=(inpar.GetNrows()-NFreeVertexPar-NExtraPar)/(NFreeTrackPar+MassOffSet-NFreeVertexPar);
  else np=(inpar.GetNrows()-NFreeVertexPar)/(NFreeTrackPar-NFreeVertexPar);
  parsize=np*(NFreeTrackPar-NFreeVertexPar)+NFreeVertexPar;
}
