/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TrackTools.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "math.h"
#include <iostream>

using namespace tauImpactParameter;

TVector3 TrackTools::propagateToXPosition(const TrackParticle& p, double x){
  double kappa=p.parameter(TrackParticle::kappa);
  double lam=p.parameter(TrackParticle::lambda);
  double phi=p.parameter(TrackParticle::phi);
  double dxy=p.parameter(TrackParticle::dxy);
  double dz=p.parameter(TrackParticle::dz);
  double r=kappa/2.0;
  double s=(asin((x+(r+dxy)*sin(phi))/r)-phi)/(2*kappa);
  double y=-r*cos(2.0*s*kappa+phi)+(r+dxy)*cos(phi);
  double z=dz+s*tan(lam);
  return TVector3(x,y,z);
}

TVector3 TrackTools::propagateToYPosition(const TrackParticle& p, double y){
  double kappa=p.parameter(TrackParticle::kappa);
  double lam=p.parameter(TrackParticle::lambda);
  double phi=p.parameter(TrackParticle::phi);
  double dxy=p.parameter(TrackParticle::dxy);
  double dz=p.parameter(TrackParticle::dz);
  double r=kappa/2.0;
  double s=(acos(((r+dxy)*cos(phi)-y)/r)-phi)/(2*kappa);
  double x=r*sin(2.0*s*kappa+phi)-(r+dxy)*sin(phi);
  double z=dz+s*tan(lam);
  return TVector3(x,y,z);
}

TVector3 TrackTools::propagateToZPosition(const TrackParticle& p, double z){
  double kappa=p.parameter(TrackParticle::kappa);
  double lam=p.parameter(TrackParticle::lambda);
  double phi=p.parameter(TrackParticle::phi);
  double dxy=p.parameter(TrackParticle::dxy);
  double dz=p.parameter(TrackParticle::dz);
  double s=(z-dz)/tan(lam);
  double r=kappa/2.0;
  double x=r*sin(2.0*s*kappa+phi)-(r+dxy)*sin(phi);
  double y=-r*cos(2.0*s*kappa+phi)+(r+dxy)*cos(phi);
  return TVector3(x,y,z);
}


LorentzVectorParticle TrackTools::lorentzParticleAtPosition(const TrackParticle& p, const TVector3& v)
{
  TVectorT<double>    FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet);
  TMatrixTSym<double> FreeParCov(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet);
  FreePar(TrackHelixVertexFitter::x0)=v.X();
  FreePar(TrackHelixVertexFitter::y0)=v.Y();
  FreePar(TrackHelixVertexFitter::z0)=v.Z();
  FreePar(TrackHelixVertexFitter::kappa0)=p.parameter(TrackParticle::kappa);
  FreePar(TrackHelixVertexFitter::lambda0)=p.parameter(TrackParticle::lambda);
  FreePar(TrackHelixVertexFitter::phi0)=p.parameter(TrackParticle::phi);
  FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::MassOffSet)=p.mass();
  FreePar(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::BField0)=p.bField();
  TVectorT<double>    LVPar=TrackHelixVertexFitter::computeLorentzVectorPar(FreePar);
  TMatrixTSym<double> LVCov=ErrorMatrixPropagator::propagateError(&TrackHelixVertexFitter::computeLorentzVectorPar,FreePar,FreeParCov);
  return LorentzVectorParticle(LVPar,LVCov,p.pdgId(),p.charge(),p.bField());
}


