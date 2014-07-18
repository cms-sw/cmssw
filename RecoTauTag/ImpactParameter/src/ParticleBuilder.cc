/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "RecoTauTag/ImpactParameter/interface/ParticleBuilder.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include <TVector3.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace tauImpactParameter;

LorentzVectorParticle ParticleBuilder::createLorentzVectorParticle(const reco::TransientTrack& transTrk, 
								   const reco::Vertex& V, bool fromPerigee,bool useTrackHelixPropagation){
  GlobalPoint p(V.position().x(),V.position().y(),V.position().z());
  TrackParticle tp=createTrackParticle(transTrk,p,fromPerigee,useTrackHelixPropagation);

  int N=TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet;
  TVectorT<double> par(N);
  par(TrackHelixVertexFitter::x0)=V.position().x();
  par(TrackHelixVertexFitter::y0)=V.position().y();
  par(TrackHelixVertexFitter::z0)=V.position().z();
  par(TrackHelixVertexFitter::kappa0)=tp.parameter(TrackParticle::kappa);
  par(TrackHelixVertexFitter::lambda0)=tp.parameter(TrackParticle::lambda);
  par(TrackHelixVertexFitter::phi0)=tp.parameter(TrackParticle::phi);
  par(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::MassOffSet)=tp.mass();
  par(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::BField0)=tp.bField();

  TMatrixTSym<double> parcov(N);
  for(int i=0;i<TrackHelixVertexFitter::NFreeVertexPar;i++){
    for(int j=0;j<TrackHelixVertexFitter::NFreeVertexPar;j++){
      parcov(i,j)=V.covariance(i,j);
    }
  }
  parcov(TrackHelixVertexFitter::kappa0,TrackHelixVertexFitter::kappa0)   = tp.covariance(TrackParticle::kappa,TrackParticle::kappa);
  parcov(TrackHelixVertexFitter::lambda0,TrackHelixVertexFitter::lambda0) = tp.covariance(TrackParticle::lambda,TrackParticle::lambda);
  parcov(TrackHelixVertexFitter::phi0,TrackHelixVertexFitter::phi0)       = tp.covariance(TrackParticle::phi,TrackParticle::phi);

  TVectorT<double>    LVPar=TrackHelixVertexFitter::computeLorentzVectorPar(par);
  TMatrixTSym<double> LVCov=ErrorMatrixPropagator::propagateError(&TrackHelixVertexFitter::computeLorentzVectorPar,par,parcov);
  return LorentzVectorParticle(LVPar,LVCov,tp.pdgId(),tp.charge(),tp.bField());
}

TrackParticle ParticleBuilder::createTrackParticle(const reco::TransientTrack& transTrk, 
						   const GlobalPoint& p, bool fromPerigee, bool useTrackHelixPropagation ){
  // Configured for CMSSW Tracks only
  TVectorT<double>    par(TrackParticle::NHelixPar+1);
  TMatrixTSym<double> cov(TrackParticle::NHelixPar+1);
  TVectorT<double>    SFpar(TrackParticle::NHelixPar);
  TMatrixTSym<double> SFcov(TrackParticle::NHelixPar);
  if(!fromPerigee){
    for(int i=0;i<TrackParticle::NHelixPar;i++){
      par(i)=transTrk.track().parameter(i);
      for(int j=0;j<TrackParticle::NHelixPar;j++){
	cov(i,j)=transTrk.track().covariance(i,j);
      }
    }
    par(TrackParticle::NHelixPar)=transTrk.field()->inInverseGeV(p).z();
    SFpar=convertCMSSWTrackParToSFTrackPar(par);
    SFcov=ErrorMatrixPropagator::propagateError(&ParticleBuilder::convertCMSSWTrackParToSFTrackPar,par,cov);
  }
  else{
    GlobalPoint TrackIPPos=transTrk.impactPointTSCP().position();
    //GlobalPoint TrackIPOrigin=transTrk.impactPointTSCP().referencePoint();
    GlobalPoint origin(0.0,0.0,0.0);
    for(int i=0;i<TrackParticle::NHelixPar;i++){
      par(i)=transTrk.trajectoryStateClosestToPoint(origin).perigeeParameters().vector()(i);
      for(int j=0;j<TrackParticle::NHelixPar;j++){
        cov(i,j)=transTrk.trajectoryStateClosestToPoint(origin).perigeeError().covarianceMatrix()(i,j);
      }
    }
    par(TrackParticle::NHelixPar)=transTrk.field()->inInverseGeV(p).z();
    SFpar=convertCMSSWTrackPerigeeToSFTrackPar(par);
    SFcov=ErrorMatrixPropagator::propagateError(&ParticleBuilder::convertCMSSWTrackPerigeeToSFTrackPar,par,cov);
    if(useTrackHelixPropagation){
      /////////////////////////////////////////////////////////////////
      // correct dxy dz neglecting material and radiative corrections
      
      LogDebug("RecoTauImpactParameterParticleBuilder") << "Offical CMS dxy - " << par(TrackParticle::dxy) << " dz " << par(TrackParticle::dz) 
		<< " kappa " << par(reco::TrackBase::i_qoverp) ;
      LogDebug("RecoTauImpactParameterParticleBuilder") << "Offical CMS dxy - SimpleFits Format" << SFpar(TrackParticle::dxy) << " dz " << SFpar(TrackParticle::dz) 
		<< " kappa " << SFpar(reco::TrackBase::i_qoverp) ;
      
      double x,y,z,dxy,dz,s,kappa,lambda,phi;
      TVectorT<double> freehelix(TrackHelixVertexFitter::NFreeTrackPar);
      freehelix(TrackHelixVertexFitter::x0)=TrackIPPos.x();
      freehelix(TrackHelixVertexFitter::y0)=TrackIPPos.y();
      freehelix(TrackHelixVertexFitter::z0)=TrackIPPos.z();
      freehelix(TrackHelixVertexFitter::kappa0)=SFpar(TrackParticle::kappa);
      freehelix(TrackHelixVertexFitter::lambda0)=SFpar(TrackParticle::lambda);
      freehelix(TrackHelixVertexFitter::phi0)=SFpar(TrackParticle::phi);
      TrackHelixVertexFitter::computedxydz(freehelix,0,kappa,lambda,phi,x,y,z,s,dxy,dz);
      SFpar(TrackParticle::dxy) = dxy;
      SFpar(TrackParticle::dz)  = dz;
      LogDebug("RecoTauImpactParameterParticleBuilder") << "Found values dxy " << dxy << " dz " << dz; 
      //exit(0);
      ////////////////////////////////////////////////////////////////
    }
  }

  PDGInfo pdgInfo;
  double c=transTrk.charge();
  return TrackParticle(SFpar,SFcov,abs(PdtPdgMini::pi_plus)*c,pdgInfo.pi_mass(),c,transTrk.field()->inInverseGeV(p).z());
}

reco::Vertex ParticleBuilder::getVertex(const LorentzVectorParticle& p){
  TVector3 v=p.vertex();
  TMatrixTSym<double> vcov=p.vertexCov();
  reco::Vertex::Point vp(v.X(),v.Y(),v.Z());
  reco::Vertex::Error ve;
  for(int i=0;i<vcov.GetNrows();i++){
    for(int j=0;j<vcov.GetNrows();j++){ve(i,j)=vcov(i,j);}
  }
  return reco::Vertex(vp,ve);
}

TVectorT<double> ParticleBuilder::convertCMSSWTrackParToSFTrackPar(const TVectorT<double>& inpar){
  TVectorT<double> par(TrackParticle::NHelixPar);
  par(TrackParticle::kappa)  = -1.0*inpar(TrackParticle::NHelixPar)*inpar(reco::TrackBase::i_qoverp)/fabs(cos(inpar(reco::TrackBase::i_lambda)));
  par(TrackParticle::lambda) = inpar(reco::TrackBase::i_lambda);
  par(TrackParticle::phi)    = inpar(reco::TrackBase::i_phi);
  par(TrackParticle::dz)     = inpar(reco::TrackBase::i_dsz)/fabs(cos(inpar(reco::TrackBase::i_lambda)));
  par(TrackParticle::dxy)    = inpar(reco::TrackBase::i_dxy);
  return par;
}

TVectorT<double> ParticleBuilder::convertCMSSWTrackPerigeeToSFTrackPar(const TVectorT<double>& inpar){
  TVectorT<double> par(TrackParticle::NHelixPar);
  par(TrackParticle::kappa)  = inpar(aCurv); 
  par(TrackParticle::lambda) = TMath::Pi()/2-inpar(aTheta); 
  par(TrackParticle::phi)    = inpar(aPhi);
  par(TrackParticle::dxy)    = -inpar(aTip);
  par(TrackParticle::dz)     = inpar(aLip);
  return par;
}



