#include <TFile.h>
#include "EgammaAnalysis/ElectronTools/interface/PFIsolationEstimator.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"

#ifndef STANDALONE
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#endif

//--------------------------------------------------------------------------------------------------
PFIsolationEstimator::PFIsolationEstimator() :
fisInitialized(kFALSE)
{
  // Constructor.
}



//--------------------------------------------------------------------------------------------------
PFIsolationEstimator::~PFIsolationEstimator()
{

}

//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initialize( Bool_t  bApplyVeto, int iParticleType ) {

  setParticleType(iParticleType);

  //By default check for an option vertex association
  checkClosestZVertex = kTRUE;
  
  //Apply vetoes
  setApplyVeto(bApplyVeto);
  
  setDeltaRVetoBarrelPhotons();
  setDeltaRVetoBarrelNeutrals();
  setDeltaRVetoBarrelCharged();
  setDeltaRVetoEndcapPhotons();
  setDeltaRVetoEndcapNeutrals();
  setDeltaRVetoEndcapCharged();

  
  setRectangleDeltaPhiVetoBarrelPhotons();
  setRectangleDeltaPhiVetoBarrelNeutrals();
  setRectangleDeltaPhiVetoBarrelCharged();
  setRectangleDeltaPhiVetoEndcapPhotons();
  setRectangleDeltaPhiVetoEndcapNeutrals();
  setRectangleDeltaPhiVetoEndcapCharged();
  

  setRectangleDeltaEtaVetoBarrelPhotons();
  setRectangleDeltaEtaVetoBarrelNeutrals();
  setRectangleDeltaEtaVetoBarrelCharged();
  setRectangleDeltaEtaVetoEndcapPhotons();
  setRectangleDeltaEtaVetoEndcapNeutrals();
  setRectangleDeltaEtaVetoEndcapCharged();


  if(bApplyVeto && iParticleType==kElectron){
    
    //Setup veto conditions for electrons
    setDeltaRVetoBarrel(kTRUE);
    setDeltaRVetoEndcap(kTRUE);
    setRectangleVetoBarrel(kFALSE);
    setRectangleVetoEndcap(kFALSE);
    setApplyDzDxyVeto(kFALSE);
    setApplyPFPUVeto(kTRUE);
    setApplyMissHitPhVeto(kTRUE); //NOTE: decided to go for this on the 26May 2012
    //Current recommended default value for the electrons
    setUseCrystalSize(kFALSE);

    // setDeltaRVetoBarrelPhotons(1E-5);   //NOTE: just to be in synch with the isoDep: fixed isoDep in 26May
    // setDeltaRVetoBarrelCharged(1E-5);    //NOTE: just to be in synch with the isoDep: fixed isoDep in 26May
    // setDeltaRVetoBarrelNeutrals(1E-5);   //NOTE: just to be in synch with the isoDep: fixed isoDep in 26May
    setDeltaRVetoEndcapPhotons(0.08);
    setDeltaRVetoEndcapCharged(0.015);
    // setDeltaRVetoEndcapNeutrals(1E-5);  //NOTE: just to be in synch with the isoDep: fixed isoDep in 26May

    setConeSize(0.4);

    
  }else{
    //Setup veto conditions for photons
    setApplyDzDxyVeto(kTRUE);
    setApplyPFPUVeto(kTRUE);
    setApplyMissHitPhVeto(kFALSE);
    setDeltaRVetoBarrel(kTRUE);
    setDeltaRVetoEndcap(kTRUE);
    setRectangleVetoBarrel(kTRUE);
    setRectangleVetoEndcap(kFALSE);
    setUseCrystalSize(kTRUE);
    setConeSize(0.3);

    setDeltaRVetoBarrelPhotons(-1);
    setDeltaRVetoBarrelNeutrals(-1);
    setDeltaRVetoBarrelCharged(0.02);
    setRectangleDeltaPhiVetoBarrelPhotons(1.);
    setRectangleDeltaPhiVetoBarrelNeutrals(-1);
    setRectangleDeltaPhiVetoBarrelCharged(-1);
    setRectangleDeltaEtaVetoBarrelPhotons(0.015);
    setRectangleDeltaEtaVetoBarrelNeutrals(-1);
    setRectangleDeltaEtaVetoBarrelCharged(-1);

    setDeltaRVetoEndcapPhotons(0.07);
    setDeltaRVetoEndcapNeutrals(-1);
    setDeltaRVetoEndcapCharged(0.02);
    setRectangleDeltaPhiVetoEndcapPhotons(-1);
    setRectangleDeltaPhiVetoEndcapNeutrals(-1);
    setRectangleDeltaPhiVetoEndcapCharged(-1);
    setRectangleDeltaEtaVetoEndcapPhotons(-1);
    setRectangleDeltaEtaVetoEndcapNeutrals(-1);
    setRectangleDeltaEtaVetoEndcapCharged(-1);
    setNumberOfCrystalEndcapPhotons(4.);
  }


}

//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initializeElectronIsolation( Bool_t  bApplyVeto){
  initialize(bApplyVeto,kElectron);
  initializeRings(1, fConeSize);

//   std::cout << " ********* Init Entering in kElectron setup "
// 	    << " bApplyVeto " << bApplyVeto
// 	    << " bDeltaRVetoBarrel " << bDeltaRVetoBarrel
// 	    << " bDeltaRVetoEndcap " << bDeltaRVetoEndcap
// 	    << " cone size " << fConeSize 
// 	    << " fDeltaRVetoEndcapPhotons " << fDeltaRVetoEndcapPhotons
// 	    << " fDeltaRVetoEndcapNeutrals " << fDeltaRVetoEndcapNeutrals
// 	    << " fDeltaRVetoEndcapCharged " << fDeltaRVetoEndcapCharged << std::endl;
  
}

//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initializePhotonIsolation( Bool_t  bApplyVeto){
  initialize(bApplyVeto,kPhoton);
  initializeRings(1, fConeSize);
}


//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initializeElectronIsolationInRings( Bool_t  bApplyVeto, int iNumberOfRings, float fRingSize ){
  initialize(bApplyVeto,kElectron);
  initializeRings(iNumberOfRings, fRingSize);
}

//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initializePhotonIsolationInRings( Bool_t  bApplyVeto, int iNumberOfRings, float fRingSize  ){
  initialize(bApplyVeto,kPhoton);
  initializeRings(iNumberOfRings, fRingSize);
}


//--------------------------------------------------------------------------------------------------
void PFIsolationEstimator::initializeRings(int iNumberOfRings, float fRingSize){
 
  setRingSize(fRingSize);
  setNumbersOfRings(iNumberOfRings);
 
  fIsolationInRings.clear();
  for(int isoBin =0;isoBin<iNumberOfRings;isoBin++){
    float fTemp = 0.0;
    fIsolationInRings.push_back(fTemp);
    
    float fTempPhoton = 0.0;
    fIsolationInRingsPhoton.push_back(fTempPhoton);

    float fTempNeutral = 0.0;
    fIsolationInRingsNeutral.push_back(fTempNeutral);

    float fTempCharged = 0.0;
    fIsolationInRingsCharged.push_back(fTempCharged);

    float fTempChargedAll = 0.0;
    fIsolationInRingsChargedAll.push_back(fTempChargedAll);

  }

  fConeSize = fRingSize * (float)iNumberOfRings;

}
  
 
//--------------------------------------------------------------------------------------------------
float PFIsolationEstimator::fGetIsolation(const reco::PFCandidate * pfCandidate, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {
 
  fGetIsolationInRings( pfCandidate, pfParticlesColl, vtx, vertices);
  refSC = reco::SuperClusterRef();
  fIsolation = fIsolationInRings[0];
  
  return fIsolation;
}


//--------------------------------------------------------------------------------------------------
std::vector<float >  PFIsolationEstimator::fGetIsolationInRings(const reco::PFCandidate * pfCandidate, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {

  int isoBin;

  
  for(isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]=0.;
    fIsolationInRingsPhoton[isoBin]=0.;
    fIsolationInRingsNeutral[isoBin]=0.;
    fIsolationInRingsCharged[isoBin]=0.;
    fIsolationInRingsChargedAll[isoBin]=0.;
  }
  
 

  fEta =  pfCandidate->eta();
  fPhi =  pfCandidate->phi();
  fPt =  pfCandidate->pt();
  fVx =  pfCandidate->vx();
  fVy =  pfCandidate->vy();
  fVz =  pfCandidate->vz();

  pivotInBarrel = fabs(pfCandidate->positionAtECALEntrance().eta())<1.479;

  for(unsigned iPF=0; iPF<pfParticlesColl->size(); iPF++) {

    const reco::PFCandidate& pfParticle= (*pfParticlesColl)[iPF]; 

    if(&pfParticle==(pfCandidate))
      continue;

    if(pfParticle.pdgId()==22){
      
      if(isPhotonParticleVetoed( &pfParticle)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsPhoton[isoBin]  = fIsolationInRingsPhoton[isoBin] + pfParticle.pt();
      }
      
    }else if(abs(pfParticle.pdgId())==130){
        
      if(isNeutralParticleVetoed(  &pfParticle)>=0.){
       	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsNeutral[isoBin]  = fIsolationInRingsNeutral[isoBin] + pfParticle.pt();
      }
    

      //}else if(abs(pfParticle.pdgId()) == 11 ||abs(pfParticle.pdgId()) == 13 || abs(pfParticle.pdgId()) == 211){
    }else if(abs(pfParticle.pdgId()) == 211){
      if(isChargedParticleVetoed( &pfParticle, vtx, vertices)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsCharged[isoBin]  = fIsolationInRingsCharged[isoBin] + pfParticle.pt();
      }

    }
  }

 
  for(int isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]= fIsolationInRingsPhoton[isoBin]+ fIsolationInRingsNeutral[isoBin] +  fIsolationInRingsCharged[isoBin];
  }

  return fIsolationInRings;
}


//--------------------------------------------------------------------------------------------------
float PFIsolationEstimator::fGetIsolation(const reco::Photon * photon, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {
 
  fGetIsolationInRings( photon, pfParticlesColl, vtx, vertices);
  fIsolation = fIsolationInRings[0];
  
  return fIsolation;
}


//--------------------------------------------------------------------------------------------------
std::vector<float >  PFIsolationEstimator::fGetIsolationInRings(const reco::Photon * photon, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {

  int isoBin;
  
  for(isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]=0.;
    fIsolationInRingsPhoton[isoBin]=0.;
    fIsolationInRingsNeutral[isoBin]=0.;
    fIsolationInRingsCharged[isoBin]=0.;
    fIsolationInRingsChargedAll[isoBin]=0.;
  }
  
  iMissHits = 0;

  refSC = photon->superCluster();
  pivotInBarrel = fabs((refSC->position().eta()))<1.479;

  for(unsigned iPF=0; iPF<pfParticlesColl->size(); iPF++) {

    const reco::PFCandidate& pfParticle= (*pfParticlesColl)[iPF]; 

    if (pfParticle.superClusterRef().isNonnull() && 
	photon->superCluster().isNonnull() && 
	pfParticle.superClusterRef() == photon->superCluster())
      continue;

    if(pfParticle.pdgId()==22){
    
      // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - pfParticle.vx(), 
  	  			    	          photon->superCluster()->y() - pfParticle.vy(), 
		    				  photon->superCluster()->z() - pfParticle.vz());

      fEta = direction.Eta();
      fPhi = direction.Phi();
      fVx  = pfParticle.vx();
      fVy  = pfParticle.vy();
      fVz  = pfParticle.vz();

      if(isPhotonParticleVetoed(&pfParticle)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsPhoton[isoBin]  = fIsolationInRingsPhoton[isoBin] + pfParticle.pt();
      }
      
    }else if(abs(pfParticle.pdgId())==130){
       
       // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - pfParticle.vx(), 
                                                  photon->superCluster()->y() - pfParticle.vy(),
                                                  photon->superCluster()->z() - pfParticle.vz());

      fEta = direction.Eta();
      fPhi = direction.Phi();
      fVx  = pfParticle.vx();
      fVy  = pfParticle.vy();
      fVz  = pfParticle.vz();
 
      if(isNeutralParticleVetoed( &pfParticle)>=0.){
       	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsNeutral[isoBin]  = fIsolationInRingsNeutral[isoBin] + pfParticle.pt();
      }

      //}else if(abs(pfParticle.pdgId()) == 11 ||abs(pfParticle.pdgId()) == 13 || abs(pfParticle.pdgId()) == 211){
    }else if(abs(pfParticle.pdgId()) == 211){
 
      // Set the vertex of reco::Photon to the first PV
      math::XYZVector direction = math::XYZVector(photon->superCluster()->x() - (*vtx).x(),
                                                  photon->superCluster()->y() - (*vtx).y(),
                                                  photon->superCluster()->z() - (*vtx).z());

      fEta = direction.Eta();
      fPhi = direction.Phi();
      fVx  = (*vtx).x();
      fVy  = (*vtx).y();
      fVz  = (*vtx).z();

      if(isChargedParticleVetoed(  &pfParticle, vtx, vertices)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsCharged[isoBin]  = fIsolationInRingsCharged[isoBin] + pfParticle.pt();
      }

    }
  }

 
  for(int isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]= fIsolationInRingsPhoton[isoBin]+ fIsolationInRingsNeutral[isoBin] +  fIsolationInRingsCharged[isoBin];
    }
  
  return fIsolationInRings;
}



//--------------------------------------------------------------------------------------------------
float PFIsolationEstimator::fGetIsolation(const reco::GsfElectron * electron, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {
 
  fGetIsolationInRings( electron, pfParticlesColl, vtx, vertices);
  fIsolation = fIsolationInRings[0];
  
  return fIsolation;
}


//--------------------------------------------------------------------------------------------------
std::vector<float >  PFIsolationEstimator::fGetIsolationInRings(const reco::GsfElectron * electron, const reco::PFCandidateCollection* pfParticlesColl,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices) {

  int isoBin;
  
  for(isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]=0.;
    fIsolationInRingsPhoton[isoBin]=0.;
    fIsolationInRingsNeutral[isoBin]=0.;
    fIsolationInRingsCharged[isoBin]=0.;
    fIsolationInRingsChargedAll[isoBin]=0.;
  }
  
  //  int iMatch =  matchPFObject(electron,pfParticlesColl);


  fEta =  electron->eta();
  fPhi =  electron->phi();
  fPt =  electron->pt();
  fVx =  electron->vx();
  fVy =  electron->vy();
  fVz =  electron->vz();
  iMissHits = electron->gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS);
  
  //  if(electron->ecalDrivenSeed())
  refSC = electron->superCluster();
  pivotInBarrel = fabs((refSC->position().eta()))<1.479;

  for(unsigned iPF=0; iPF<pfParticlesColl->size(); iPF++) {

    const reco::PFCandidate& pfParticle= (*pfParticlesColl)[iPF]; 
 
 
    if(pfParticle.pdgId()==22){
    
      if(isPhotonParticleVetoed(&pfParticle)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsPhoton[isoBin]  = fIsolationInRingsPhoton[isoBin] + pfParticle.pt();

      }
      
    }else if(abs(pfParticle.pdgId())==130){
        
      if(isNeutralParticleVetoed( &pfParticle)>=0.){
       	isoBin = (int)(fDeltaR/fRingSize);
	fIsolationInRingsNeutral[isoBin]  = fIsolationInRingsNeutral[isoBin] + pfParticle.pt();
      }

      //}else if(abs(pfParticle.pdgId()) == 11 ||abs(pfParticle.pdgId()) == 13 || abs(pfParticle.pdgId()) == 211){
    }else if(abs(pfParticle.pdgId()) == 211){
      if(isChargedParticleVetoed(  &pfParticle, vtx, vertices)>=0.){
	isoBin = (int)(fDeltaR/fRingSize);
	
	fIsolationInRingsCharged[isoBin]  = fIsolationInRingsCharged[isoBin] + pfParticle.pt();
      }

    }
  }

 
  for(int isoBin =0;isoBin<iNumberOfRings;isoBin++){
    fIsolationInRings[isoBin]= fIsolationInRingsPhoton[isoBin]+ fIsolationInRingsNeutral[isoBin] +  fIsolationInRingsCharged[isoBin];
    }
  
  return fIsolationInRings;
}


//--------------------------------------------------------------------------------------------------
float  PFIsolationEstimator::isPhotonParticleVetoed( const reco::PFCandidate* pfIsoCand ){
  
  
  fDeltaR = deltaR(fEta,fPhi,pfIsoCand->eta(),pfIsoCand->phi()); 

  if(fDeltaR > fConeSize)
    return -999.;
  
  fDeltaPhi = deltaPhi(fPhi,pfIsoCand->phi()); 
  fDeltaEta = fEta-pfIsoCand->eta(); 

  if(!bApplyVeto)
    return fDeltaR;
 
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  //      this will be changed in the future
  
  if(bApplyMissHitPhVeto) {
    if(iMissHits > 0)
      if(pfIsoCand->mva_nothing_gamma() > 0.99) {
	if(pfIsoCand->superClusterRef().isNonnull() && refSC.isNonnull()) {
	  if(pfIsoCand->superClusterRef() == refSC)
	    return -999.;
	}
      }
  }

  if(pivotInBarrel){
    if(bDeltaRVetoBarrel){
      if(fDeltaR < fDeltaRVetoBarrelPhotons)
        return -999.;
    }
    
    if(bRectangleVetoBarrel){
      if(abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelPhotons && abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelPhotons){
	return -999.;
      }
    }
  }else{
    if (bUseCrystalSize == true) {
      fDeltaRVetoEndcapPhotons = 0.00864*fabs(sinh(refSC->position().eta()))*fNumberOfCrystalEndcapPhotons;
    }

    if(bDeltaRVetoEndcap){
      if(fDeltaR < fDeltaRVetoEndcapPhotons)
	return -999.;
    }
    if(bRectangleVetoEndcap){
      if(abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapPhotons && abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapPhotons){
	 return -999.;
      }
    }
  }

  return fDeltaR;
}

//--------------------------------------------------------------------------------------------------
float  PFIsolationEstimator::isNeutralParticleVetoed( const reco::PFCandidate* pfIsoCand ){

  fDeltaR = deltaR(fEta,fPhi,pfIsoCand->eta(),pfIsoCand->phi()); 
  
  if(fDeltaR > fConeSize)
    return -999;
  
  fDeltaPhi = deltaPhi(fPhi,pfIsoCand->phi()); 
  fDeltaEta = fEta-pfIsoCand->eta(); 

  if(!bApplyVeto)
    return fDeltaR;

  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  //      this will be changed in the future
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel&&!bRectangleVetoBarrel){
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel){
	if(fDeltaR < fDeltaRVetoBarrelNeutrals)
	  return -999.;
      }
      if(bRectangleVetoBarrel){
	if(abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelNeutrals && abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelNeutrals){
	    return -999.;
	}
      }
      
    }else{
     if(!bDeltaRVetoEndcap&&!bRectangleVetoEndcap){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap){
	if(fDeltaR < fDeltaRVetoEndcapNeutrals)
	  return -999.;
      }
      if(bRectangleVetoEndcap){
	if(abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapNeutrals && abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapNeutrals){
	  return -999.;
	}
      }
  }

  return fDeltaR;
}


//----------------------------------------------------------------------------------------------------
float  PFIsolationEstimator::isChargedParticleVetoed(const reco::PFCandidate* pfIsoCand, edm::Handle< reco::VertexCollection >  vertices  ){
  //need code to handle special conditions
  
  return -999;
}

//-----------------------------------------------------------------------------------------------------
float  PFIsolationEstimator::isChargedParticleVetoed(const reco::PFCandidate* pfIsoCand,reco::VertexRef vtxMain, edm::Handle< reco::VertexCollection >  vertices  ){
  
  reco::VertexRef vtx = chargedHadronVertex(vertices,  *pfIsoCand );
  if(vtx.isNull())
    return -999.;
  
//   float fVtxMainX = (*vtxMain).x();
//   float fVtxMainY = (*vtxMain).y();
  float fVtxMainZ = (*vtxMain).z();

  if(bApplyPFPUVeto) {
    if(vtx != vtxMain)
      return -999.;
  }
    

  if(bApplyDzDxyVeto) {
    if(iParticleType==kPhoton){
      
      float dz = fabs( pfIsoCand->trackRef()->dz( (*vtxMain).position() ) );
      if (dz > 0.2)
        return -999.;
	
      double dxy = pfIsoCand->trackRef()->dxy( (*vtxMain).position() );  
      if (fabs(dxy) > 0.1)
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
      
      
      float dz = fabs(vtx->z() - fVtxMainZ);
      if (dz > 1.)
	return -999.;
      
      double dxy = ( -(vtx->x() - fVx)*pfIsoCand->py() + (vtx->y() - fVy)*pfIsoCand->px()) / pfIsoCand->pt();
      if(fabs(dxy) > 0.1)
	return -999.;
    }
  }    

  fDeltaR = deltaR(pfIsoCand->eta(),pfIsoCand->phi(),fEta,fPhi); 

  if(fDeltaR > fConeSize)
    return -999.;

  fDeltaPhi = deltaPhi(fPhi,pfIsoCand->phi()); 
  fDeltaEta = fEta-pfIsoCand->eta(); 
  

//   std::cout << " charged hadron: DR " <<  fDeltaR 
// 	    << " pt " <<  pfIsoCand->pt() << " eta,phi " << pfIsoCand->eta() << ", " << pfIsoCand->phi()
// 	    << " fVtxMainZ " << (*vtxMain).z() << " cand z " << vtx->z() << std::endl;
  

  if(!bApplyVeto)
    return fDeltaR;  
  
  //NOTE: get the direction for the EB/EE transition region from the deposit just to be in synch with the isoDep
  //      this will be changed in the future  
  if(pivotInBarrel){
    if(!bDeltaRVetoBarrel&&!bRectangleVetoBarrel){
      return fDeltaR;
    }
    
    if(bDeltaRVetoBarrel){
	if(fDeltaR < fDeltaRVetoBarrelCharged)
	  return -999.;
      }
      if(bRectangleVetoBarrel){
	if(abs(fDeltaEta) < fRectangleDeltaEtaVetoBarrelCharged && abs(fDeltaPhi) < fRectangleDeltaPhiVetoBarrelCharged){
	    return -999.;
	}
      }
      
    }else{
     if(!bDeltaRVetoEndcap&&!bRectangleVetoEndcap){
       return fDeltaR;
     }
      if(bDeltaRVetoEndcap){
	if(fDeltaR < fDeltaRVetoEndcapCharged)
	  return -999.;
      }
      if(bRectangleVetoEndcap){
	if(abs(fDeltaEta) < fRectangleDeltaEtaVetoEndcapCharged && abs(fDeltaPhi) < fRectangleDeltaPhiVetoEndcapCharged){
	  return -999.;
	}
      }
  }
		   
  


  return fDeltaR;
}


//--------------------------------------------------------------------------------------------------
reco::VertexRef  PFIsolationEstimator::chargedHadronVertex(  edm::Handle< reco::VertexCollection > verticesColl, const reco::PFCandidate& pfcand ){

  //code copied from Florian's PFNoPU class
    
  reco::TrackBaseRef trackBaseRef( pfcand.trackRef() );

  size_t  iVertex = 0;
  unsigned index=0;
  unsigned nFoundVertex = 0;

  float bestweight=0;
  
  const reco::VertexCollection& vertices = *(verticesColl.product());

  for( reco::VertexCollection::const_iterator iv=vertices.begin(); iv!=vertices.end(); ++iv, ++index) {
    
    const reco::Vertex& vtx = *iv;
    
    // loop on tracks in vertices
    for(reco::Vertex::trackRef_iterator iTrack=vtx.tracks_begin();iTrack!=vtx.tracks_end(); ++iTrack) {

      const reco::TrackBaseRef& baseRef = *iTrack;

      // one of the tracks in the vertex is the same as 
      // the track considered in the function
      if(baseRef == trackBaseRef ) {
        float w = vtx.trackWeight(baseRef);
        //select the vertex for which the track has the highest weight
        if (w > bestweight){
          bestweight=w;
          iVertex=index;
          nFoundVertex++;
        }
      }
    }
    
  }
 
 
  
  if (nFoundVertex>0){
    if (nFoundVertex!=1)
      edm::LogWarning("TrackOnTwoVertex")<<"a track is shared by at least two verteces. Used to be an assert";
    return  reco::VertexRef( verticesColl, iVertex);
  }
  // no vertex found with this track. 

  // optional: as a secondary solution, associate the closest vertex in z
  if ( checkClosestZVertex ) {

    double dzmin = 10000.;
    double ztrack = pfcand.vertex().z();
    bool foundVertex = false;
    index = 0;
    for( reco::VertexCollection::const_iterator  iv=vertices.begin(); iv!=vertices.end(); ++iv, ++index) {

      double dz = fabs(ztrack - iv->z());
      if(dz<dzmin) {
        dzmin = dz;
        iVertex = index;
        foundVertex = true;
      }
    }

    if( foundVertex ) 
      return  reco::VertexRef( verticesColl, iVertex);  
  
  }
   
  return  reco::VertexRef( );
}



int PFIsolationEstimator::matchPFObject(const reco::Photon* photon, const reco::PFCandidateCollection* Candidates ){
  
  Int_t iMatch = -1;

  int i=0;
  for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
    const reco::PFCandidate& pfParticle = (*iPF);
    //    if((((pfParticle.pdgId()==22 && pfParticle.mva_nothing_gamma()>0.01) || TMath::Abs(pfParticle.pdgId())==11) )){
    if((((pfParticle.pdgId()==22 ) || TMath::Abs(pfParticle.pdgId())==11) )){
     
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





int PFIsolationEstimator::matchPFObject(const reco::GsfElectron* electron, const reco::PFCandidateCollection* Candidates ){
  
  Int_t iMatch = -1;

  int i=0;
  for(reco::PFCandidateCollection::const_iterator iPF=Candidates->begin();iPF !=Candidates->end();iPF++){
    const reco::PFCandidate& pfParticle = (*iPF);
    //    if((((pfParticle.pdgId()==22 && pfParticle.mva_nothing_gamma()>0.01) || TMath::Abs(pfParticle.pdgId())==11) )){
    if((((pfParticle.pdgId()==22 ) || TMath::Abs(pfParticle.pdgId())==11) )){
     
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
      if((((pfParticle.pdgId()==22 ) || TMath::Abs(pfParticle.pdgId())==11) )){
	if(pfParticle.pt()>fPt){
	  fDeltaR = deltaR(pfParticle.eta(),pfParticle.phi(),electron->eta(),electron->phi());
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

