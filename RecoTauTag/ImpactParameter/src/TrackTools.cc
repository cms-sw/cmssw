/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TrackTools.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "math.h"
#include <iostream>

TVector3 TrackTools::PropogateToXPosition(TrackParticle &p,double &x){
  double kappa=p.Parameter(TrackParticle::kappa);
  double lam=p.Parameter(TrackParticle::lambda);
  double phi=p.Parameter(TrackParticle::phi);
  double dxy=p.Parameter(TrackParticle::dxy);
  double dz=p.Parameter(TrackParticle::dz);
  double r=kappa/2.0;
  double s=(asin((x+(r+dxy)*sin(phi))/r)-phi)/(2*kappa);
  double y=-r*cos(2.0*s*kappa+phi)+(r+dxy)*cos(phi);
  double z=dz+s*tan(lam);
  return TVector3(x,y,z);
}

TVector3 TrackTools::PropogateToYPosition(TrackParticle &p,double &y){
  double kappa=p.Parameter(TrackParticle::kappa);
  double lam=p.Parameter(TrackParticle::lambda);
  double phi=p.Parameter(TrackParticle::phi);
  double dxy=p.Parameter(TrackParticle::dxy);
  double dz=p.Parameter(TrackParticle::dz);
  double r=kappa/2.0;
  double s=(acos(((r+dxy)*cos(phi)-y)/r)-phi)/(2*kappa);
  double x=r*sin(2.0*s*kappa+phi)-(r+dxy)*sin(phi);
  double z=dz+s*tan(lam);
  return TVector3(x,y,z);
}

TVector3 TrackTools::PropogateToZPosition(TrackParticle &p,double &z){
  double kappa=p.Parameter(TrackParticle::kappa);
  double lam=p.Parameter(TrackParticle::lambda);
  double phi=p.Parameter(TrackParticle::phi);
  double dxy=p.Parameter(TrackParticle::dxy);
  double dz=p.Parameter(TrackParticle::dz);
  double s=(z-dz)/tan(lam);
  double r=kappa/2.0;
  double x=r*sin(2.0*s*kappa+phi)-(r+dxy)*sin(phi);
  double y=-r*cos(2.0*s*kappa+phi)+(r+dxy)*cos(phi);
  return TVector3(x,y,z);
}


LorentzVectorParticle TrackTools::LorentzParticleAtPosition(TrackParticle &p,TVector3 &v){
  TMatrixT<double>    FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet,1);
  TMatrixTSym<double> FreeParCov(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet);
  FreePar(TrackHelixVertexFitter::x0,0)=v.X();
  FreePar(TrackHelixVertexFitter::y0,0)=v.Y();
  FreePar(TrackHelixVertexFitter::z0,0)=v.Z();
  FreePar(TrackHelixVertexFitter::kappa0,0)=p.Parameter(TrackParticle::kappa);
  FreePar(TrackHelixVertexFitter::lambda0,0)=p.Parameter(TrackParticle::lambda);
  FreePar(TrackHelixVertexFitter::phi0,0)=p.Parameter(TrackParticle::phi);
  FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::MassOffSet,0)=p.Mass();
  FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::BField0,0)=p.BField();
  TMatrixT<double>    LVPar=TrackHelixVertexFitter::ComputeLorentzVectorPar(FreePar);
  TMatrixTSym<double> LVCov=ErrorMatrixPropagator::PropogateError(&TrackHelixVertexFitter::ComputeLorentzVectorPar,FreePar,FreeParCov);
  return LorentzVectorParticle(LVPar,LVCov,p.PDGID(),p.Charge(),p.BField());
}


