/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "TDecompBK.h"
#include <iostream>

TrackHelixVertexFitter::TrackHelixVertexFitter(std::vector<TrackParticle> &particles_,TVector3 vguess):
  isFit(false),
  isConfigure(false),
  nParticles(particles_.size()),
  nPar((NFreeTrackPar-NFreeVertexPar)*particles_.size()+NFreeVertexPar),
  nVal(TrackParticle::NHelixPar*particles_.size())
{
  particles=particles_;
  par.ResizeTo(nPar,1);
  parcov.ResizeTo(nPar,nPar);
  val.ResizeTo(nVal,1);
  cov.ResizeTo(nVal,nVal);
  for(unsigned int p=0; p<particles.size();p++){
    for(unsigned int j=0; j<TrackParticle::NHelixPar;j++){
      val(MeasuredValueIndex(j,p),0)=particles.at(p).Parameter(j);
      for(unsigned int k=0; k<TrackParticle::NHelixPar;k++){
	cov(MeasuredValueIndex(j,p),MeasuredValueIndex(k,p))=particles.at(p).Covariance(j,k);
      }
    }
  }
  TDecompBK Inverter(cov);
  double det = cov.Determinant();
  if(!Inverter.Decompose()){
    std::cout << "TrackHelixVertexFitter::TrackHelixVertexFitter Fit failed: unable to invert SYM gain matrix " << det << " \n" << std::endl;
    return;
  }


  cov_inv.ResizeTo(nVal,nVal);
  cov_inv=Inverter.Invert();
  ndf=nVal-nPar;
  // Set Initial conditions within reason
  par(x0,0)  = vguess.X(); parcov(x0,x0)=pow(1.0,2.0);
  par(y0,0)  = vguess.Y(); parcov(y0,y0)=pow(1.0,2.0);
  par(z0,0)  = vguess.Z(); parcov(z0,z0)=pow(1.0,2.0);
  for(unsigned int p=0; p<particles.size();p++){
    par(FreeParIndex(kappa0,p),0)  = val(MeasuredValueIndex(TrackParticle::kappa,p),0);
    par(FreeParIndex(lambda0,p),0) = val(MeasuredValueIndex(TrackParticle::lambda,p),0);
    par(FreeParIndex(phi0,p),0)    = val(MeasuredValueIndex(TrackParticle::phi,p),0);
    //    
    parcov(FreeParIndex(kappa0,p),FreeParIndex(kappa0,p))   = cov(MeasuredValueIndex(TrackParticle::kappa,p),MeasuredValueIndex(TrackParticle::kappa,p));
    parcov(FreeParIndex(lambda0,p),FreeParIndex(lambda0,p)) = cov(MeasuredValueIndex(TrackParticle::lambda,p),MeasuredValueIndex(TrackParticle::lambda,p));
    parcov(FreeParIndex(phi0,p),FreeParIndex(phi0,p))       = cov(MeasuredValueIndex(TrackParticle::phi,p),MeasuredValueIndex(TrackParticle::phi,p));
  }
  isConfigure=true;
  ////////////////////////////////////////////////////////////////
  // debug statements
  /*
  for(int i=0;i<val.GetNrows();i++) std::cout << "input Val " << val(i,0) << " " << TrackParticle::Name(i%TrackParticle::NHelixPar) << std::endl;
  for(int i=0;i<cov.GetNrows();i++){
    for(int j=0;j<cov.GetNrows();j++)  std::cout << cov(i,j) << " ";
    std::cout << std::endl;
  }
  for(int i=0;i<par.GetNrows();i++) std::cout << "input Par " << par(i,0) << " " <<  FreeParName(i) << std::endl;
  for(int i=0;i<parcov.GetNrows();i++){
    for(int j=0;j<parcov.GetNrows();j++)  std::cout << parcov(i,j) << " ";
    std::cout << std::endl;
    }*/
  ////////////////////////////////////////////////////////////////
}

TrackHelixVertexFitter::~TrackHelixVertexFitter(){}

double TrackHelixVertexFitter::UpdateChisquare(TMatrixT<double> inpar){
  TMatrixT<double> vprime=ComputePar(inpar);
  TMatrixT<double> dalpha=vprime-val;
  TMatrixT<double> dalphaT=dalpha;  dalphaT.T();
  TMatrixT<double> chisquare=dalphaT*(cov_inv*dalpha);
  return chisquare(0,0);
}

std::vector<TrackParticle> TrackHelixVertexFitter::GetReFitTracks(){
  std::vector<TrackParticle> refitParticles;
  for(unsigned int p=0;p<particles.size();p++){
    TMatrixT<double> FreePar(NFreeTrackPar,1);
    TMatrixTSym<double> FreeParCov(NFreeTrackPar);
    for(int i=0;i<FreeParCov.GetNrows();i++){
      FreePar(i,0)=par(FreeParIndex(i,p),0);
      for(int j=0;j<FreeParCov.GetNrows();j++){
	FreeParCov(i,j)=parcov(FreeParIndex(i,p),FreeParIndex(j,p));
      }
    }
    TMatrixT<double>    TrackPar=ComputePar(FreePar);
    TMatrixTSym<double> TrackCov=ErrorMatrixPropagator::PropogateError(&TrackHelixVertexFitter::ComputePar,FreePar,FreeParCov);
    refitParticles.push_back(TrackParticle(TrackPar,TrackCov,particles.at(p).PDGID(),particles.at(p).Mass(),particles.at(p).Charge(),particles.at(p).BField()));
  }
  return particles;
}



std::vector<LorentzVectorParticle> TrackHelixVertexFitter::GetReFitLorentzVectorParticles(){
  std::vector<LorentzVectorParticle> refitParticles;
  for(unsigned int p=0;p<particles.size();p++){
    TMatrixT<double>    FreePar(NFreeTrackPar+NExtraPar+MassOffSet,1);
    TMatrixTSym<double> FreeParCov(NFreeTrackPar+NExtraPar+MassOffSet);
    for(int i=0;i<NFreeTrackPar;i++){
      FreePar(i,0)=par(FreeParIndex(i,p),0);
      for(int j=0;j<NFreeTrackPar;j++){
        FreeParCov(i,j)=parcov(FreeParIndex(i,p),FreeParIndex(j,p));
      }
    }
    FreePar(NFreeTrackPar+MassOffSet,0)=particles.at(p).Mass();
    FreePar(NFreeTrackPar+BField0,0)=particles.at(p).BField();
    TMatrixT<double>    LVPar=ComputeLorentzVectorPar(FreePar);
    TMatrixTSym<double> LVCov=ErrorMatrixPropagator::PropogateError(&TrackHelixVertexFitter::ComputeLorentzVectorPar,FreePar,FreeParCov);
    refitParticles.push_back(LorentzVectorParticle(LVPar,LVCov,particles.at(p).PDGID(),particles.at(p).Charge(),particles.at(p).BField()));
  }
  return refitParticles;
}

LorentzVectorParticle TrackHelixVertexFitter::GetMother(int pdgid){
  double c(0),b(0);
  TMatrixT<double>    FreePar(par.GetNrows()+NExtraPar+particles.size(),1);
  TMatrixTSym<double> FreeParCov(par.GetNrows()+NExtraPar+particles.size());
  for(int i=0;i<par.GetNrows();i++){
    FreePar(i,0)=par(i,0);
    for(int j=0;j<par.GetNrows();j++){FreeParCov(i,j)=parcov(i,j);}
  }
  for(unsigned int p=0; p<particles.size();p++){
    b=particles.at(p).BField();
    c+=particles.at(p).Charge();
    FreePar(par.GetNrows()+MassOffSet+p,0)=particles.at(p).Mass();
  }
  FreePar(par.GetNrows()+BField0,0)=b;
  TMatrixT<double>    mpar=ComputeMotherLorentzVectorPar(FreePar);
  TMatrixTSym<double> mcov=ErrorMatrixPropagator::PropogateError(&TrackHelixVertexFitter::ComputeMotherLorentzVectorPar,FreePar,FreeParCov);
  return LorentzVectorParticle(mpar,mcov,pdgid,c,b);
}

TVector3 TrackHelixVertexFitter::GetVertex(){
  return TVector3(par(FreeParIndex(x0,0),0),par(FreeParIndex(y0,0),0),par(FreeParIndex(z0,0),0));
}

TMatrixTSym<double> TrackHelixVertexFitter::GetVertexError(){
  TMatrixTSym<double> c(NFreeVertexPar);
  for(unsigned int i=0;i<NFreeVertexPar;i++){
    for(unsigned int j=0;j<NFreeVertexPar;j++){c(FreeParIndex(i,0),FreeParIndex(j,0))=parcov(FreeParIndex(i,0),FreeParIndex(j,0));}
  }
  return c;
}

void TrackHelixVertexFitter::Computedxydz(TMatrixT<double> &inpar,int p,double &kappa,double &lam,double &phi,double &x,double &y,double &z,double &s,double &dxy,double &dz){
  kappa=inpar(FreeParIndex(kappa0,p),0);
  lam=inpar(FreeParIndex(lambda0,p),0);
  phi=inpar(FreeParIndex(phi0,p),0);
  x=inpar(FreeParIndex(x0,p),0);
  y=inpar(FreeParIndex(y0,p),0);
  z=inpar(FreeParIndex(z0,p),0);
  double v=(2.0*kappa*(x*cos(phi)+y*sin(phi)));
  double arcsinv=0;
  //std::cout << "v " << v << std::endl;
  if(v>=1.0){arcsinv=TMath::Pi()/2;}
  else if(v<=-1.0){arcsinv=-TMath::Pi()/2;}
  else{arcsinv=asin(v);}
  s=1.0/(2.0*kappa)*arcsinv;//asin(2.0*kappa*(x*cos(phi)+y*sin(phi)));
  dxy=y*cos(phi)-x*sin(phi)-(1/kappa)*sin(kappa*s)*sin(kappa*s);
  dz=z-s*tan(lam);
  ///////////////////////////////
  // debug
  /*  
  std::cout << "kappa0 "   << inpar(FreeParIndex(kappa0,p),0)
            << " lambda0 " << inpar(FreeParIndex(lambda0,p),0)
            << " phi0 "    << inpar(FreeParIndex(phi0,p),0)
            << "  x0 "     << inpar(FreeParIndex(x0,p),0)
            << "  y0 "     << inpar(FreeParIndex(y0,p),0)
            << "  z0 "     << inpar(FreeParIndex(z0,p),0) << std::endl;
  std::cout << "arcsin " << asin(2*kappa*(x*cos(phi)+y*sin(phi))) << " c " << kappa << " cosphi " << cos(phi) << " sinphi " << sin(phi) << " F " << x*cos(phi)+y*sin(phi) << " s " << s << std::endl; 
  std::cout << "kappa " << kappa << " lam " << lam << " phi " << phi << " x " << x << " y " << y << " z " << z << " s " << s <<  " v " << v << " dxy " << dxy << " dz " << dz << std::endl;
  std::cout << "TrackHelixVertexFitter::Computedxydz done" << std::endl;
  */
}

TMatrixT<double> TrackHelixVertexFitter::ComputePar(TMatrixT<double> &inpar){
  int nparticles=(inpar.GetNrows()-NFreeVertexPar)/(NFreeTrackPar-NFreeVertexPar);
  TMatrixT<double> helices(nparticles*TrackParticle::NHelixPar,1);
  for(int p=0;p<nparticles;p++){
    TMatrixT<double> TrackPar=ComputeTrackPar(inpar,p);
    for(int i=0;i<TrackParticle::NHelixPar;i++){helices(MeasuredValueIndex(i,p),0)=TrackPar(i,0);}
  }
  return helices;
}

TMatrixT<double> TrackHelixVertexFitter::ComputeTrackPar(TMatrixT<double> &inpar, int p){
  TMatrixT<double> helix(TrackParticle::NHelixPar,1);
  // copy parameters that are 1 to 1
  double kappa,lam,phi,x,y,z,s,dxy,dz;
  TrackHelixVertexFitter::Computedxydz(inpar,p,kappa,lam,phi,x,y,z,s,dxy,dz);
  helix(TrackParticle::kappa,0)=kappa;
  helix(TrackParticle::lambda,0)=lam;
  helix(TrackParticle::phi,0)=phi;
  helix(TrackParticle::dxy,0)=dxy;
  helix(TrackParticle::dz,0)=dz;
  return helix;
}

TMatrixT<double> TrackHelixVertexFitter::ComputeLorentzVectorPar(TMatrixT<double> &inpar){
  int np(0), parsize(0); ParSizeInfo(inpar,np,parsize,true);
  double B=inpar(parsize+BField0,0);
  double massHypothesis=inpar(parsize+MassOffSet,0);
  TMatrixT<double> LV(LorentzVectorParticle::NLorentzandVertexPar,1);
  double kappa,lam,phi,x,y,z,s,dxy,dz;
  int p=0;
  TrackHelixVertexFitter::Computedxydz(inpar,p,kappa,lam,phi,x,y,z,s,dxy,dz);
  LV(LorentzVectorParticle::px,0)=B*(1.0/fabs(kappa))*cos(2*s*kappa+phi);
  LV(LorentzVectorParticle::py,0)=B*(1.0/fabs(kappa))*sin(2*s*kappa+phi);
  LV(LorentzVectorParticle::pz,0)=B*(1.0/fabs(kappa))*tan(lam) ;
  LV(LorentzVectorParticle::m,0) =massHypothesis;
  LV(LorentzVectorParticle::vx,0)=x;
  LV(LorentzVectorParticle::vy,0)=y;
  LV(LorentzVectorParticle::vz,0)=z;
  return LV;
}

TMatrixT<double> TrackHelixVertexFitter::ComputeMotherLorentzVectorPar(TMatrixT<double> &inpar){
  TMatrixT<double> mother(LorentzVectorParticle::NLorentzandVertexPar,1);
  double E(0);
  int np(0), parsize(0); ParSizeInfo(inpar,np,parsize,true);
  for(int p=0;p<np;p++){
    TMatrixT<double> particlepar(NFreeTrackPar+NExtraPar+MassOffSet,1);
    for(int i=0;i<NFreeTrackPar;i++){particlepar(i,0)=inpar(FreeParIndex(i,p),0);}
    particlepar(NFreeTrackPar+BField0,0)=inpar(parsize+BField0,0);
    particlepar(NFreeTrackPar+MassOffSet,0)=inpar(parsize+MassOffSet+p,0);
    TMatrixT<double> daughter=TrackHelixVertexFitter::ComputeLorentzVectorPar(particlepar);
    mother(LorentzVectorParticle::px,0)+=daughter(LorentzVectorParticle::px,0);
    mother(LorentzVectorParticle::py,0)+=daughter(LorentzVectorParticle::py,0);
    mother(LorentzVectorParticle::pz,0)+=daughter(LorentzVectorParticle::pz,0);
    mother(LorentzVectorParticle::vx,0)=daughter(LorentzVectorParticle::vx,0);
    mother(LorentzVectorParticle::vy,0)=daughter(LorentzVectorParticle::vy,0);
    mother(LorentzVectorParticle::vz,0)=daughter(LorentzVectorParticle::vz,0);
    E+=sqrt((daughter(LorentzVectorParticle::px,0)*daughter(LorentzVectorParticle::px,0)+
	     daughter(LorentzVectorParticle::py,0)*daughter(LorentzVectorParticle::py,0)+
	     daughter(LorentzVectorParticle::pz,0)*daughter(LorentzVectorParticle::pz,0)+
	     daughter(LorentzVectorParticle::m,0)*daughter(LorentzVectorParticle::m,0)));
  }
  double P2=(mother(LorentzVectorParticle::px,0)*mother(LorentzVectorParticle::px,0)+
	     mother(LorentzVectorParticle::py,0)*mother(LorentzVectorParticle::py,0)+
	     mother(LorentzVectorParticle::pz,0)*mother(LorentzVectorParticle::pz,0));
  mother(LorentzVectorParticle::m,0)=(E*E-P2)/sqrt(fabs(E*E-P2));
  return mother;
}

TString TrackHelixVertexFitter::FreeParName(int Par){
  int p(0);
  if(Par==x0)     return "x0";
  if(Par==y0)     return "y0";
  if(Par==z0)     return "z0";
  for(p=0;p<nParticles;p++){
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

void TrackHelixVertexFitter::ParSizeInfo(TMatrixT<double> &inpar, int &np, int &parsize, bool hasextras){
  if(hasextras)np=(inpar.GetNrows()-NFreeVertexPar-NExtraPar)/(NFreeTrackPar+MassOffSet-NFreeVertexPar);
  else np=(inpar.GetNrows()-NFreeVertexPar)/(NFreeTrackPar-NFreeVertexPar);
  parsize=np*(NFreeTrackPar-NFreeVertexPar)+NFreeVertexPar;
}
