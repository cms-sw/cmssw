/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/ErrorMatrixPropagator.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include "RecoTauTag/ImpactParameter/interface/ParticleBuilder.h"
#include "RecoTauTag/ImpactParameter/interface/ParticleMassHelper.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"
#include <TVector3.h>

LorentzVectorParticle ParticleBuilder::CreateLorentzVectorParticle(reco::TransientTrack &transTrk, edm::ESHandle<TransientTrackBuilder>  &transTrackBuilder,reco::Vertex &V, bool fromPerigee,bool useTrackHelixPropogation){
  GlobalPoint p(V.position().x(),V.position().y(),V.position().z());
  TrackParticle tp=CreateTrackParticle(transTrk,transTrackBuilder,p,fromPerigee,useTrackHelixPropogation);

  int N=TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::NExtraPar+TrackHelixVertexFitter::MassOffSet;
  TMatrixT<double> par(N,1);
  par(TrackHelixVertexFitter::x0,0)=V.position().x();
  par(TrackHelixVertexFitter::y0,0)=V.position().y();
  par(TrackHelixVertexFitter::z0,0)=V.position().z();
  par(TrackHelixVertexFitter::kappa0,0)=tp.Parameter(TrackParticle::kappa);
  par(TrackHelixVertexFitter::lambda0,0)=tp.Parameter(TrackParticle::lambda);
  par(TrackHelixVertexFitter::phi0,0)=tp.Parameter(TrackParticle::phi);
  par(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::MassOffSet,0)=tp.Mass();
  par(TrackHelixVertexFitter::NFreeTrackPar+TrackHelixVertexFitter::BField0,0)=tp.BField();

  TMatrixTSym<double> parcov(N);
  for(int i=0;i<TrackHelixVertexFitter::NFreeVertexPar;i++){
    for(int j=0;j<TrackHelixVertexFitter::NFreeVertexPar;j++){
      parcov(i,j)=V.covariance(i,j);
    }
  }
  parcov(TrackHelixVertexFitter::kappa0,TrackHelixVertexFitter::kappa0)   = tp.Covariance(TrackParticle::kappa,TrackParticle::kappa);
  parcov(TrackHelixVertexFitter::lambda0,TrackHelixVertexFitter::lambda0) = tp.Covariance(TrackParticle::lambda,TrackParticle::lambda);
  parcov(TrackHelixVertexFitter::phi0,TrackHelixVertexFitter::phi0)       = tp.Covariance(TrackParticle::phi,TrackParticle::phi);

  TMatrixT<double>    LVPar=TrackHelixVertexFitter::ComputeLorentzVectorPar(par);
  TMatrixTSym<double> LVCov=ErrorMatrixPropagator::PropogateError(&TrackHelixVertexFitter::ComputeLorentzVectorPar,par,parcov);
  return LorentzVectorParticle(LVPar,LVCov,tp.PDGID(),tp.Charge(),tp.BField());
}


TrackParticle ParticleBuilder::CreateTrackParticle(reco::TransientTrack &transTrk, edm::ESHandle<TransientTrackBuilder>  &transTrackBuilder, const GlobalPoint p, bool fromPerigee,bool useTrackHelixPropogation){
  // Configured for CMSSW Tracks only
  TMatrixT<double>    par(TrackParticle::NHelixPar+1,1);
  TMatrixTSym<double> cov(TrackParticle::NHelixPar+1);
  TMatrixT<double>    SFpar(TrackParticle::NHelixPar,1);
  TMatrixTSym<double> SFcov(TrackParticle::NHelixPar);
  if(!fromPerigee){
    for(int i=0;i<TrackParticle::NHelixPar;i++){
      par(i,0)=transTrk.track().parameter(i);
      for(int j=0;j<TrackParticle::NHelixPar;j++){
	cov(i,j)=transTrk.track().covariance(i,j);
      }
    }
    par(TrackParticle::NHelixPar,0)=transTrackBuilder->field()->inInverseGeV(p).z();
    SFpar=ConvertCMSSWTrackParToSFTrackPar(par);
    SFcov=ErrorMatrixPropagator::PropogateError(&ParticleBuilder::ConvertCMSSWTrackParToSFTrackPar,par,cov);
  }
  else{
    GlobalPoint TrackIPPos=transTrk.impactPointTSCP().position();
    //GlobalPoint TrackIPOrigin=transTrk.impactPointTSCP().referencePoint();
    GlobalPoint origin(0.0,0.0,0.0);
    for(int i=0;i<TrackParticle::NHelixPar;i++){
      par(i,0)=transTrk.trajectoryStateClosestToPoint(origin).perigeeParameters().vector()(i);
      for(int j=0;j<TrackParticle::NHelixPar;j++){
        cov(i,j)=transTrk.trajectoryStateClosestToPoint(origin).perigeeError().covarianceMatrix()(i,j);
      }
    }
    par(TrackParticle::NHelixPar,0)=transTrackBuilder->field()->inInverseGeV(p).z();
    SFpar=ConvertCMSSWTrackPerigeeToSFTrackPar(par);
    SFcov=ErrorMatrixPropagator::PropogateError(&ParticleBuilder::ConvertCMSSWTrackPerigeeToSFTrackPar,par,cov);
    if(useTrackHelixPropogation){
      /////////////////////////////////////////////////////////////////
      // correct dxy dz neglecting material and radiative corrections
      /*
      std::cout << "Offical CMS dxy - " << par(TrackParticle::dxy,0) << " dz " << par(TrackParticle::dz,0) 
		<< " kappa " <<  track->qoverp() << " " << par(reco::TrackBase::i_qoverp,0) <<  std::endl;
      std::cout << "Offical CMS dxy - SimpleFits Format" << SFpar(TrackParticle::dxy,0) << " dz " << SFpar(TrackParticle::dz,0) 
		<< " kappa " <<  track->qoverp() << " " << SFpar(reco::TrackBase::i_qoverp,0) <<  std::endl;
      std::cout << "x " << TrackIPOrigin.x() << " y " <<  TrackIPOrigin.y() << " z " <<  TrackIPOrigin.z() << std::endl;
      */
      double x,y,z,dxy,dz,s,kappa,lambda,phi;
      TMatrixT<double>    freehelix(TrackHelixVertexFitter::NFreeTrackPar,1);
      freehelix(TrackHelixVertexFitter::x0,0)=TrackIPPos.x();
      freehelix(TrackHelixVertexFitter::y0,0)=TrackIPPos.y();
      freehelix(TrackHelixVertexFitter::z0,0)=TrackIPPos.z();
      freehelix(TrackHelixVertexFitter::kappa0,0)=SFpar(TrackParticle::kappa,0);
      freehelix(TrackHelixVertexFitter::lambda0,0)=SFpar(TrackParticle::lambda,0);
      freehelix(TrackHelixVertexFitter::phi0,0)=SFpar(TrackParticle::phi,0);
      TrackHelixVertexFitter::Computedxydz(freehelix,0,kappa,lambda,phi,x,y,z,s,dxy,dz);
      SFpar(TrackParticle::dxy,0) = dxy;
      SFpar(TrackParticle::dz,0)  = dz;
      //std::cout << "Found values dxy " << dxy << " dz " << dz << std::endl; 
      //exit(0);
      ////////////////////////////////////////////////////////////////
    }
  }

  ParticleMassHelper PMH;
  double c=transTrk.charge();
  return  TrackParticle(SFpar,SFcov,abs(PdtPdgMini::pi_plus)*c,PMH.Get_piMass(),c,transTrackBuilder->field()->inInverseGeV(p).z());
}

reco::Vertex ParticleBuilder::GetVertex(LorentzVectorParticle p){
  TVector3 v=p.Vertex();
  TMatrixTSym<double> vcov=p.VertexCov();
  reco::Vertex::Point vp(v.X(),v.Y(),v.Z());
  reco::Vertex::Error ve;
  for(int i=0;i<vcov.GetNrows();i++){
    for(int j=0;j<vcov.GetNrows();j++){ve(i,j)=vcov(i,j);}
  }
  return reco::Vertex(vp,ve);
}


TMatrixT<double> ParticleBuilder::ConvertCMSSWTrackParToSFTrackPar(TMatrixT<double> &inpar){
  TMatrixT<double> par(TrackParticle::NHelixPar,1);
  par(TrackParticle::kappa,0)  = -1.0*inpar(TrackParticle::NHelixPar,0)*inpar(reco::TrackBase::i_qoverp,0)/fabs(cos(inpar(reco::TrackBase::i_lambda,0)));
  par(TrackParticle::lambda,0) = inpar(reco::TrackBase::i_lambda,0);
  par(TrackParticle::phi,0)    = inpar(reco::TrackBase::i_phi,0);
  par(TrackParticle::dz,0)     = inpar(reco::TrackBase::i_dsz,0)/fabs(cos(inpar(reco::TrackBase::i_lambda,0)));
  par(TrackParticle::dxy,0)    = inpar(reco::TrackBase::i_dxy,0);
  return par;
}


TMatrixT<double> ParticleBuilder::ConvertCMSSWTrackPerigeeToSFTrackPar(TMatrixT<double> &inpar){
  TMatrixT<double> par(TrackParticle::NHelixPar,1);
  par(TrackParticle::kappa,0)  = inpar(aCurv,0); 
  par(TrackParticle::lambda,0) = TMath::Pi()/2-inpar(aTheta,0); 
  par(TrackParticle::phi,0)    = inpar(aPhi,0);
  par(TrackParticle::dxy,0)    = -inpar(aTip,0);
  par(TrackParticle::dz,0)     = inpar(aLip,0);
  return par;
}



