#include "RecoEgamma/PhotonIdentification/interface/PFPhotonIsolationCalculator.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"



#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"



void PFPhotonIsolationCalculator::setup(const edm::ParameterSet& conf) {
 

 iParticleType_ = conf.getParameter<int>("particleType");
  if (  iParticleType_ ==1 ) { 

    fConeSize_     = conf.getParameter<double>("coneDR");
    iNumberOfRings_ = conf.getParameter<int>("numberOfRings");
    fRingSize_     = conf.getParameter<double>("ringSize");
    //
    bApplyVeto_    = conf.getParameter<bool>("applyVeto");
    bApplyPFPUVeto_    = conf.getParameter<bool>("applyPFPUVeto");
    bApplyDzDxyVeto_    = conf.getParameter<bool>("applyDzDxyVeto");
    bApplyMissHitPhVeto_    = conf.getParameter<bool>("applyMissHitPhVeto");
    bDeltaRVetoBarrel_    = conf.getParameter<bool>("deltaRVetoBarrel");
    bDeltaRVetoEndcap_    = conf.getParameter<bool>("deltaRVetoEndcap");
    bRectangleVetoBarrel_    = conf.getParameter<bool>("rectangleVetoBarrel");
    bRectangleVetoEndcap_    = conf.getParameter<bool>("rectangleVetoEndcap");
    bUseCrystalSize_    = conf.getParameter<bool>("useCrystalSize");
    //
    fDeltaRVetoBarrelPhotons_    = conf.getParameter<double>("deltaRVetoBarrelPhotons");   
    fDeltaRVetoBarrelNeutrals_    = conf.getParameter<double>("deltaRVetoBarrelNeutrals");   
    fDeltaRVetoBarrelCharged_    = conf.getParameter<double>("deltaRVetoBarrelCharged");   
    fDeltaRVetoEndcapPhotons_    = conf.getParameter<double>("deltaRVetoEndcapPhotons");   
    fDeltaRVetoEndcapNeutrals_    = conf.getParameter<double>("deltaRVetoEndcapNeutrals");   
    fDeltaRVetoEndcapCharged_    = conf.getParameter<double>("deltaRVetoEndcapCharged");   
    fNumberOfCrystalEndcapPhotons_ = conf.getParameter<double>("numberOfCrystalEndcapPhotons");  
    //
    fRectangleDeltaPhiVetoBarrelPhotons_    = conf.getParameter<double>("rectangleDeltaPhiVetoBarrelPhotons");
    fRectangleDeltaPhiVetoBarrelNeutrals_   = conf.getParameter<double>("rectangleDeltaPhiVetoBarrelNeutrals");
    fRectangleDeltaPhiVetoBarrelCharged_    = conf.getParameter<double>("rectangleDeltaPhiVetoBarrelCharged");
    fRectangleDeltaPhiVetoEndcapPhotons_    = conf.getParameter<double>("rectangleDeltaPhiVetoEndcapPhotons");
    fRectangleDeltaPhiVetoEndcapNeutrals_   = conf.getParameter<double>("rectangleDeltaPhiVetoEndcapNeutrals");
    fRectangleDeltaPhiVetoEndcapCharged_    = conf.getParameter<double>("rectangleDeltaPhiVetoEndcapCharged");
    //
    fRectangleDeltaEtaVetoBarrelPhotons_    = conf.getParameter<double>("rectangleDeltaEtaVetoBarrelPhotons");
    fRectangleDeltaEtaVetoBarrelNeutrals_   = conf.getParameter<double>("rectangleDeltaEtaVetoBarrelNeutrals");
    fRectangleDeltaEtaVetoBarrelCharged_    = conf.getParameter<double>("rectangleDeltaEtaVetoBarrelCharged");
    fRectangleDeltaEtaVetoEndcapPhotons_    = conf.getParameter<double>("rectangleDeltaEtaVetoEndcapPhotons");
    fRectangleDeltaEtaVetoEndcapNeutrals_   = conf.getParameter<double>("rectangleDeltaEtaVetoEndcapNeutrals");
    fRectangleDeltaEtaVetoEndcapCharged_    = conf.getParameter<double>("rectangleDeltaEtaVetoEndcapCharged");
    //
    bCheckClosestZVertex_    = conf.getParameter<bool>("checkClosestZVertex");
    initializeRings(iNumberOfRings_, fConeSize_);
    
  }  
 



}



//--------------------------------------------------------------------------------------------------

PFPhotonIsolationCalculator::PFPhotonIsolationCalculator() {
  // Constructor.
}



//--------------------------------------------------------------------------------------------------
PFPhotonIsolationCalculator::~PFPhotonIsolationCalculator()
{

}


void PFPhotonIsolationCalculator::calculate(const reco::Photon* aPho, 
					    const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
					    //					    reco::VertexRef vtx,
					    edm::Handle< reco::VertexCollection >& vertices,
					    const edm::Event& e , const edm::EventSetup& es,
					    reco::Photon::PflowIsolationVariables& phoisol03) {

  reco::VertexRef vtx(vertices, 0);
  this->fGetIsolation(&*aPho,  pfCandidateHandle, vtx, vertices);
  phoisol03.chargedHadronIso = this->getIsolationCharged() ;
  phoisol03.neutralHadronIso = this->getIsolationNeutral();
  phoisol03.photonIso        = this->getIsolationPhoton();


  
 }




//--------------------------------------------------------------------------------------------------
void PFPhotonIsolationCalculator::initializeRings(int iNumberOfRings, float fRingSize){
 
  fIsolationInRings_.clear();
  for(int isoBin =0;isoBin<iNumberOfRings;isoBin++){
    float fTemp = 0.0;
    fIsolationInRings_.push_back(fTemp);
    
    float fTempPhoton = 0.0;
    fIsolationInRingsPhoton_.push_back(fTempPhoton);

    float fTempNeutral = 0.0;
    fIsolationInRingsNeutral_.push_back(fTempNeutral);

    float fTempCharged = 0.0;
    fIsolationInRingsCharged_.push_back(fTempCharged);

    float fTempChargedAll = 0.0;
    fIsolationInRingsChargedAll_.push_back(fTempChargedAll);

  }

  fConeSize_ = fRingSize * (float)iNumberOfRings;

}
  
 


//--------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::fGetIsolation(const reco::Photon * photon, const  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle ,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {
 
  fGetIsolationInRings( photon, pfCandidateHandle, vtx, vertices);
  fIsolation_ = fIsolationInRings_[0];
  return fIsolation_;
}



//--------------------------------------------------------------------------------------------------
std::vector<float > PFPhotonIsolationCalculator::fGetIsolationInRings(const reco::Photon * photon, const  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle, reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {


  int isoBin;
  
  for(isoBin =0;isoBin<iNumberOfRings_;isoBin++){
    fIsolationInRings_[isoBin]=0.;
    fIsolationInRingsPhoton_[isoBin]=0.;
    fIsolationInRingsNeutral_[isoBin]=0.;
    fIsolationInRingsCharged_[isoBin]=0.;
    fIsolationInRingsChargedAll_[isoBin]=0.;
  }
  
  iMissHits_ = 0;

  refSC = photon->superCluster();
  pivotInBarrel = std::fabs((refSC->position().eta()))<1.479;

  for(unsigned iPF=0; iPF< pfCandidateHandle->size(); iPF++) {

    reco::PFCandidateRef pfParticle(reco::PFCandidateRef(pfCandidateHandle, iPF));

    if (pfParticle->superClusterRef().isNonnull() &&
        photon->superCluster().isNonnull() &&
        pfParticle->superClusterRef() == photon->superCluster()) 
      continue;
    

  
    if(pfParticle->pdgId()==22){
      // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - pfParticle->vx(),
						  photon->superCluster()->y() - pfParticle->vy(),
                                                  photon->superCluster()->z() - pfParticle->vz());
      
      fEta_ = direction.Eta();
      fPhi_ = direction.Phi();
      fVx_ = pfParticle->vx();
      fVy_ = pfParticle->vy();
      fVz_ = pfParticle->vz();

      float fDeltaR =isPhotonParticleVetoed(pfParticle); 
      if( fDeltaR >=0.){
        isoBin = (int)(fDeltaR/fRingSize_);
        fIsolationInRingsPhoton_[isoBin] = fIsolationInRingsPhoton_[isoBin] + pfParticle->pt();
      }
      
    }else if(std::abs(pfParticle->pdgId())==130){
      // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - pfParticle->vx(),
                                                  photon->superCluster()->y() - pfParticle->vy(),
                                                  photon->superCluster()->z() - pfParticle->vz());

      fEta_ = direction.Eta();
      fPhi_ = direction.Phi();
      fVx_ = pfParticle->vx();
      fVy_ = pfParticle->vy();
      fVz_ = pfParticle->vz();
      float fDeltaR =  isNeutralParticleVetoed( pfParticle); 
      if( fDeltaR>=0.){
	isoBin = (int)(fDeltaR/fRingSize_);
        fIsolationInRingsNeutral_[isoBin] = fIsolationInRingsNeutral_[isoBin] + pfParticle->pt();
      }

      //}else if(abs(pfParticle.pdgId()) == 11 ||abs(pfParticle.pdgId()) == 13 || abs(pfParticle.pdgId()) == 211){
    }else if(std::abs(pfParticle->pdgId()) == 211){
      // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - (*vtx).x(),
                                                  photon->superCluster()->y() - (*vtx).y(),
                                                  photon->superCluster()->z() - (*vtx).z());

      fEta_ = direction.Eta();
      fPhi_ = direction.Phi();
      fVx_ = (*vtx).x();
      fVy_ = (*vtx).y();
      fVz_ = (*vtx).z();
      float fDeltaR = isChargedParticleVetoed( pfParticle, vtx, vertices);
      if( fDeltaR >=0.){
        isoBin = (int)(fDeltaR/fRingSize_);
        fIsolationInRingsCharged_[isoBin] = fIsolationInRingsCharged_[isoBin] + pfParticle->pt();
      }

    }
  }

 
  for(int isoBin =0;isoBin<iNumberOfRings_;isoBin++){
    fIsolationInRings_[isoBin]= fIsolationInRingsPhoton_[isoBin]+ fIsolationInRingsNeutral_[isoBin] + fIsolationInRingsCharged_[isoBin];
  }
  
  return fIsolationInRings_;
}




//--------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isPhotonParticleVetoed( const reco::PFCandidate* pfIsoCand ){
  
  
  float fDeltaR = deltaR(fEta_,fPhi_,pfIsoCand->eta(),pfIsoCand->phi());

  if(fDeltaR > fConeSize_)
    return -999.;
  
  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();

  if(!bApplyVeto_)
    return fDeltaR;
 
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  
  if(bApplyMissHitPhVeto_) {
    if(iMissHits_ > 0)
      if(pfIsoCand->mva_nothing_gamma() > 0.99) {
        if(pfIsoCand->superClusterRef().isNonnull() && refSC.isNonnull()) {
         if(pfIsoCand->superClusterRef() == refSC)
         return -999.;
        }
      }
  }

  if(pivotInBarrel){
    if(bDeltaRVetoBarrel_){
      if(fDeltaR < fDeltaRVetoBarrelPhotons_)
        return -999.;
    }
    
    if(bRectangleVetoBarrel_){
      if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelPhotons_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelPhotons_ ){
        return -999.;
      }
    }
  }else{
    if (bUseCrystalSize_ == true) {
      fDeltaRVetoEndcapPhotons_ = 0.00864*std::fabs(refSC->position().z()/sqrt(refSC->position().perp2()))*fNumberOfCrystalEndcapPhotons_;
    }

    if(bDeltaRVetoEndcap_){
      if(fDeltaR < fDeltaRVetoEndcapPhotons_)
        return -999.;
    }
    if(bRectangleVetoEndcap_){
      if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapPhotons_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapPhotons_ ){
         return -999.;
      }
    }
  }

  return fDeltaR;
}



//--------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isPhotonParticleVetoed( const reco::PFCandidateRef pfIsoCand ){
  
  
  float fDeltaR = deltaR(fEta_,fPhi_,pfIsoCand->eta(),pfIsoCand->phi());

  if(fDeltaR > fConeSize_)
    return -999.;
  
  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();

  if(!bApplyVeto_)
    return fDeltaR;
 
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  
  if(bApplyMissHitPhVeto_) {
    if(iMissHits_ > 0)
      if(pfIsoCand->mva_nothing_gamma() > 0.99) {
        if(pfIsoCand->superClusterRef().isNonnull() && refSC.isNonnull()) {
         if(pfIsoCand->superClusterRef() == refSC)
         return -999.;
        }
      }
  }

  if(pivotInBarrel){
    if(bDeltaRVetoBarrel_){
      if(fDeltaR < fDeltaRVetoBarrelPhotons_)
        return -999.;
    }
    
    if(bRectangleVetoBarrel_){
      if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelPhotons_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelPhotons_){
        return -999.;
      }
    }
  }else{
    if (bUseCrystalSize_ == true) {
      fDeltaRVetoEndcapPhotons_ = 0.00864*std::fabs(refSC->position().z()/sqrt(refSC->position().perp2()))*fNumberOfCrystalEndcapPhotons_;
    }

    if(bDeltaRVetoEndcap_){
      if(fDeltaR < fDeltaRVetoEndcapPhotons_)
        return -999.;
    }
    if(bRectangleVetoEndcap_){
      if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapPhotons_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapPhotons_){
         return -999.;
      }
    }
  }

  return fDeltaR;
}


//--------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isNeutralParticleVetoed( const reco::PFCandidate* pfIsoCand ){

  float fDeltaR = deltaR(fEta_,fPhi_,pfIsoCand->eta(),pfIsoCand->phi());
  
  if(fDeltaR > fConeSize_)
    return -999;
  
  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();

  if(!bApplyVeto_)
    return fDeltaR;

  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel_ && !bRectangleVetoBarrel_){
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel_ ){
        if(fDeltaR < fDeltaRVetoBarrelNeutrals_)
         return -999.;
      }
      if(bRectangleVetoBarrel_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelNeutrals_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelNeutrals_){
         return -999.;
        }
      }
      
    }else{
     if(!bDeltaRVetoEndcap_  &&!  bRectangleVetoEndcap_){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap_){
        if(fDeltaR < fDeltaRVetoEndcapNeutrals_)
         return -999.;
      }
      if(bRectangleVetoEndcap_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapNeutrals_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapNeutrals_){
         return -999.;
        }
      }
  }

  return fDeltaR;
}


//--------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isNeutralParticleVetoed( const reco::PFCandidateRef pfIsoCand ){

  float fDeltaR = deltaR(fEta_,fPhi_,pfIsoCand->eta(),pfIsoCand->phi());
  
  if(fDeltaR > fConeSize_)
    return -999;
  
  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();

  if(!bApplyVeto_)
    return fDeltaR;

  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel_  && !bRectangleVetoBarrel_ ){
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel_){
        if(fDeltaR < fDeltaRVetoBarrelNeutrals_)
         return -999.;
      }
      if(bRectangleVetoBarrel_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelNeutrals_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelNeutrals_){
         return -999.;
        }
      }
      
    }else{
     if(!bDeltaRVetoEndcap_ && !bRectangleVetoEndcap_ ){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap_ ){
        if(fDeltaR < fDeltaRVetoEndcapNeutrals_ )
         return -999.;
      }
      if(bRectangleVetoEndcap_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapNeutrals_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapNeutrals_){
         return -999.;
        }
      }
  }

  return fDeltaR;
}




//----------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isChargedParticleVetoed(const reco::PFCandidate* pfIsoCand, edm::Handle< reco::VertexCollection > vertices ){
  //need code to handle special conditions
  
  return -999;
}

//-----------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isChargedParticleVetoed(const reco::PFCandidate* pfIsoCand,reco::VertexRef vtxMain, edm::Handle< reco::VertexCollection > vertices ){
  
  reco::VertexRef vtx = chargedHadronVertex(vertices, *pfIsoCand );
  if(vtx.isNull())
    return -999.;
  
// float fVtxMainX = (*vtxMain).x();
// float fVtxMainY = (*vtxMain).y();
  float fVtxMainZ = (*vtxMain).z();

  if(bApplyPFPUVeto_) {
    if(vtx != vtxMain)
      return -999.;
  }
    

  if(bApplyDzDxyVeto_) {
    if(iParticleType_==kPhoton){
      
      float dz = std::fabs( pfIsoCand->trackRef()->dz( (*vtxMain).position() ) );
      if (dz > 0.2)
        return -999.;
        
      double dxy = pfIsoCand->trackRef()->dxy( (*vtxMain).position() );
      if (std::fabs(dxy) > 0.1)
        return -999.;
      
      /*
float dz = fabs(vtx->z() - fVtxMainZ);
if (dz > 1.)
        return -999.;
double dxy = ( -(vtx->x() - fVtxMainX)*pfIsoCand->py() + (vtx->y() - fVtxMainY)*pfIsoCand->px()) / pfIsoCand->pt();
if(fabs(dxy) > 0.2)
        return -999.;
*/
    }else{
      
      
      float dz = std::fabs(vtx->z() - fVtxMainZ);
      if (dz > 1.)
        return -999.;
      
      double dxy = ( -(vtx->x() - fVx_)*pfIsoCand->py() + (vtx->y() - fVy_)*pfIsoCand->px()) / pfIsoCand->pt();
      if(std::fabs(dxy) > 0.1)
        return -999.;
    }
  }

  float fDeltaR = deltaR(pfIsoCand->eta(),pfIsoCand->phi(),fEta_,fPhi_);

  if(fDeltaR > fConeSize_)
    return -999.;

  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();
  

// std::abscout << " charged hadron: DR " << fDeltaR
//          << " pt " << pfIsoCand->pt() << " eta,phi " << pfIsoCand->eta() << ", " << pfIsoCand->phi()
//          << " fVtxMainZ " << (*vtxMain).z() << " cand z " << vtx->z() << std::endl;
  

  if(!bApplyVeto_)
    return fDeltaR;
  
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel_ &&!bRectangleVetoBarrel_ ){
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel_){
        if(fDeltaR < fDeltaRVetoBarrelCharged_)
         return -999.;
      }
      if(bRectangleVetoBarrel_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelCharged_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelCharged_){
         return -999.;
        }
      }
      
    }else{
     if(!bDeltaRVetoEndcap_  && !bRectangleVetoEndcap_){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap_){
        if(fDeltaR < fDeltaRVetoEndcapCharged_)
         return -999.;
      }
      if(bRectangleVetoEndcap_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapCharged_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapCharged_){
         return -999.;
        }
      }
  }
                
  


  return fDeltaR;
}


//-----------------------------------------------------------------------------------------------------
float PFPhotonIsolationCalculator::isChargedParticleVetoed(const reco::PFCandidateRef pfIsoCand,reco::VertexRef vtxMain, edm::Handle< reco::VertexCollection > vertices ){
  
  reco::VertexRef vtx = chargedHadronVertex(vertices, *pfIsoCand );
  if(vtx.isNull())
    return -999.;
  
// float fVtxMainX = (*vtxMain).x();
// float fVtxMainY = (*vtxMain).y();
  float fVtxMainZ = (*vtxMain).z();

  if(bApplyPFPUVeto_) {
    if(vtx != vtxMain)
      return -999.;
  }
    

  if(bApplyDzDxyVeto_) {
    if(iParticleType_==kPhoton){
      
      float dz = std::fabs( pfIsoCand->trackRef()->dz( (*vtxMain).position() ) );
      if (dz > 0.2)
        return -999.;
        
      double dxy = pfIsoCand->trackRef()->dxy( (*vtxMain).position() );
      if (std::fabs(dxy) > 0.1)
        return -999.;
      
      /*
float dz = fabs(vtx->z() - fVtxMainZ);
if (dz > 1.)
        return -999.;
double dxy = ( -(vtx->x() - fVtxMainX)*pfIsoCand->py() + (vtx->y() - fVtxMainY)*pfIsoCand->px()) / pfIsoCand->pt();
if(fabs(dxy) > 0.2)
        return -999.;
*/
    }else{
      
      
      float dz = std::fabs(vtx->z() - fVtxMainZ);
      if (dz > 1.)
        return -999.;
      
      double dxy = ( -(vtx->x() - fVx_)*pfIsoCand->py() + (vtx->y() - fVy_)*pfIsoCand->px()) / pfIsoCand->pt();
      if(std::fabs(dxy) > 0.1)
        return -999.;
    }
  }

  float fDeltaR = deltaR(pfIsoCand->eta(),pfIsoCand->phi(),fEta_,fPhi_);

  if(fDeltaR > fConeSize_)
    return -999.;

  float fDeltaPhi = deltaPhi(fPhi_,pfIsoCand->phi());
  float fDeltaEta = fEta_-pfIsoCand->eta();
  

// std::cout << " charged hadron: DR " << fDeltaR
//          << " pt " << pfIsoCand->pt() << " eta,phi " << pfIsoCand->eta() << ", " << pfIsoCand->phi()
//          << " fVtxMainZ " << (*vtxMain).z() << " cand z " << vtx->z() << std::endl;
  

  if(!bApplyVeto_)
    return fDeltaR;
  
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  // this will be changed in the future
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel_ && !bRectangleVetoBarrel_) {
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel_){
        if(fDeltaR < fDeltaRVetoBarrelCharged_)
         return -999.;
      }
      if(bRectangleVetoBarrel_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelCharged_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelCharged_){
         return -999.;
        }
      }
      
    }else{
     if(!bDeltaRVetoEndcap_ &&!bRectangleVetoEndcap_ ){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap_){
        if(fDeltaR < fDeltaRVetoEndcapCharged_)
         return -999.;
      }
      if(bRectangleVetoEndcap_){
        if(std::abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapCharged_ && std::abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapCharged_){
         return -999.;
        }
      }
  }
                
  


  return fDeltaR;
}



//--------------------------------------------------------------------------------------------------
reco::VertexRef PFPhotonIsolationCalculator::chargedHadronVertex( edm::Handle< reco::VertexCollection > verticesColl, const reco::PFCandidate& pfcand ){

  //code copied from Florian's PFNoPU class (corrected removing the double loop....)
    
  auto const & track = pfcand.trackRef();

  size_t iVertex = 0;
  unsigned int index=0;
  unsigned int nFoundVertex = 0;

  float bestweight=0;
  
  const reco::VertexCollection& vertices = *(verticesColl.product());

  for( auto const & vtx :  vertices) {
    float w = vtx.trackWeight(track); // 0 if does not belong here
    //select the vertex for which the track has the highest weight
    if (w > bestweight){ // should we break here?
        bestweight=w;
	iVertex=index;
	nFoundVertex++;
    }
    ++index; 
  }
 
 
  
  if (nFoundVertex>0){
    if (nFoundVertex!=1)
      edm::LogWarning("TrackOnTwoVertex")<<"a track is shared by at least two verteces. Used to be an assert";
    return reco::VertexRef( verticesColl, iVertex);
  }
  // no vertex found with this track.

  // optional: as a secondary solution, associate the closest vertex in z
  if (  bCheckClosestZVertex_ ) {

    double dzmin = 10000.;
    double ztrack = pfcand.vertex().z();
    bool foundVertex = false;
    index = 0;
    for( reco::VertexCollection::const_iterator iv=vertices.begin(); iv!=vertices.end(); ++iv, ++index) {

      double dz = std::fabs(ztrack - iv->z());
      if(dz<dzmin) {
        dzmin = dz;
        iVertex = index;
        foundVertex = true;
      }
    }

    if( foundVertex )
      return reco::VertexRef( verticesColl, iVertex);
  
  }
   
  return reco::VertexRef( );
}



int PFPhotonIsolationCalculator::matchPFObject(const reco::Photon* photon, const reco::PFCandidateCollection* Candidates ){
  
  Int_t iMatch = -1;

  int i=0;
  for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
    const reco::PFCandidate& pfParticle = (*iPF);
    if((((pfParticle.pdgId()==22 ) || std::abs(pfParticle.pdgId())==11) )){
     
      if(pfParticle.superClusterRef()==photon->superCluster())
        iMatch= i;
     
    }
    
    i++;
  }
  
/*
if(iMatch == -1){
i=0;
float fPt = -1;
for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
const reco::PFCandidate& pfParticle = (*iPF);
if((((pfParticle.pdgId()==22 ) || TMath::Abs(pfParticle.pdgId())==11) )){
        if(pfParticle.pt()>fPt){
         fDeltaR = deltaR(pfParticle.eta(),pfParticle.phi(),photon->eta(),photon->phi());
         if(fDeltaR<0.1){
         iMatch = i;
         fPt = pfParticle.pt();
         }
        }
}
i++;
}
}
*/

  return iMatch;

}





int PFPhotonIsolationCalculator::matchPFObject(const reco::GsfElectron* electron, const reco::PFCandidateCollection* Candidates ){
  
  Int_t iMatch = -1;

  int i=0;
  for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
    const reco::PFCandidate& pfParticle = (*iPF);
    if((((pfParticle.pdgId()==22 ) || std::abs(pfParticle.pdgId())==11) )){
     
      if(pfParticle.superClusterRef()==electron->superCluster())
        iMatch= i;
     
    }
    
    i++;
  }
  
  if(iMatch == -1){
    i=0;
    float fPt = -1;
    for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
      const reco::PFCandidate& pfParticle = (*iPF);
      if((((pfParticle.pdgId()==22 ) || std::abs(pfParticle.pdgId())==11) )){
        if(pfParticle.pt()>fPt){
         float fDeltaR = deltaR(pfParticle.eta(),pfParticle.phi(),electron->eta(),electron->phi());
         if(fDeltaR<0.1){
         iMatch = i;
         fPt = pfParticle.pt();
         }
        }
      }
      i++;
    }
  }
  
  return iMatch;

}

