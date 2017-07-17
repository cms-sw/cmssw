//--------------------------------------------------------------------------------------------------
// $Id $
//
// PFPhotonIsolationCalculator
//
// Class for calculating PFIsolation for Photons.
//This class takes
// PF Particle collection and the reconstructed vertex collection as input.
//
// Authors: Vasundhara Chetluru
// Modified specifically for Photons, to be used as configurable plug-in 
// algorithm in the gedPhotonProducer, by N. Marinelli
//--------------------------------------------------------------------------------------------------


#ifndef PFPhotonIsolationCalculator_H
#define PFPhotonIsolationCalculator_H


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"



class PFPhotonIsolationCalculator{
 public:
  PFPhotonIsolationCalculator();
  ~PFPhotonIsolationCalculator();


  void setup(const edm::ParameterSet& conf);
  


 public:

  

  void calculate(const reco::Photon*, 
		 const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
		 edm::Handle< reco::VertexCollection >& vertices,
		 const edm::Event& e, const edm::EventSetup& es,
		 reco::Photon::PflowIsolationVariables& phoisol03) ;


  
 
 private:


 enum VetoType {
    kElectron = -1, // MVA for non-triggering electrons
    kPhoton = 1 // MVA for triggering electrons
  };
 

 void initializeRings(int iNumberOfRings, float fRingSize);
 
 
 float getIsolationPhoton(){ fIsolationPhoton_ =  fIsolationInRingsPhoton_[0]; return fIsolationPhoton_; };
 float getIsolationNeutral(){ fIsolationNeutral_ = fIsolationInRingsNeutral_[0]; return fIsolationNeutral_; };
 float getIsolationCharged(){ fIsolationCharged_ = fIsolationInRingsCharged_[0]; return fIsolationCharged_; };
 float getIsolationChargedAll(){ return fIsolationChargedAll_; };
 
 std::vector<float > getIsolationInRingsPhoton(){ return fIsolationInRingsPhoton_; };
 std::vector<float > getIsolationInRingsNeutral(){ return fIsolationInRingsNeutral_; };
 std::vector<float > getIsolationInRingsCharged(){ return fIsolationInRingsCharged_; };
 std::vector<float > getIsolationInRingsChargedAll(){ return fIsolationInRingsChargedAll_; };


  //Veto implementation
  float isPhotonParticleVetoed( const  reco::PFCandidateRef pfIsoCand );
  float isPhotonParticleVetoed( const reco::PFCandidate* pfIsoCand );
  //
  float isNeutralParticleVetoed( const reco::PFCandidate* pfIsoCand );
  float isNeutralParticleVetoed( const reco::PFCandidateRef pfIsoCand );
  //
  float isChargedParticleVetoed( const reco::PFCandidate* pfIsoCand, edm::Handle< reco::VertexCollection > vertices);
  float isChargedParticleVetoed( const reco::PFCandidate* pfIsoCand,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices );
  float isChargedParticleVetoed( const reco::PFCandidateRef pfIsoCand,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices );
 


  reco::VertexRef chargedHadronVertex(edm::Handle< reco::VertexCollection > verticies, const reco::PFCandidate& pfcand );

  int matchPFObject(const reco::Photon* photon, const reco::PFCandidateCollection* pfParticlesColl );
  int matchPFObject(const reco::GsfElectron* photon, const reco::PFCandidateCollection* pfParticlesColl );

  float fGetIsolation(const reco::Photon* photon, const  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle ,reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices );
  std::vector<float > fGetIsolationInRings(const reco::Photon* photon,  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle, reco::VertexRef vtx, edm::Handle< reco::VertexCollection > vertices);
    

  int iParticleType_;

  Bool_t fisInitialized_;
  float fIsolation_;
  float fIsolationPhoton_;
  float fIsolationNeutral_;
  float fIsolationCharged_;
  float fIsolationChargedAll_;
  
  std::vector<float > fIsolationInRings_;
  std::vector<float > fIsolationInRingsPhoton_;
  std::vector<float > fIsolationInRingsNeutral_;
  std::vector<float > fIsolationInRingsCharged_;
  std::vector<float > fIsolationInRingsChargedAll_;

  Bool_t bCheckClosestZVertex_;
  float  fConeSize_;
  Bool_t bApplyVeto_;
  Bool_t bApplyDzDxyVeto_;
  Bool_t bApplyPFPUVeto_;
  Bool_t bApplyMissHitPhVeto_;
  Bool_t bUseCrystalSize_;

  Bool_t bDeltaRVetoBarrel_;
  Bool_t bDeltaRVetoEndcap_;
  
  Bool_t bRectangleVetoBarrel_;
  Bool_t bRectangleVetoEndcap_;
  
  float fDeltaRVetoBarrelPhotons_;
  float fDeltaRVetoBarrelNeutrals_;
  float fDeltaRVetoBarrelCharged_;

  float fDeltaRVetoEndcapPhotons_;
  float fDeltaRVetoEndcapNeutrals_;
  float fDeltaRVetoEndcapCharged_;

  float fNumberOfCrystalEndcapPhotons_;

  float fRectangleDeltaPhiVetoBarrelPhotons_;
  float fRectangleDeltaPhiVetoBarrelNeutrals_;
  float fRectangleDeltaPhiVetoBarrelCharged_;

  float fRectangleDeltaPhiVetoEndcapPhotons_;
  float fRectangleDeltaPhiVetoEndcapNeutrals_;
  float fRectangleDeltaPhiVetoEndcapCharged_;
  
  float fRectangleDeltaEtaVetoBarrelPhotons_;
  float fRectangleDeltaEtaVetoBarrelNeutrals_;
  float fRectangleDeltaEtaVetoBarrelCharged_;

  float fRectangleDeltaEtaVetoEndcapPhotons_;
  float fRectangleDeltaEtaVetoEndcapNeutrals_;
  float fRectangleDeltaEtaVetoEndcapCharged_;

  int iNumberOfRings_;
  int iMissHits_;

  float fRingSize_;
  //  float fDeltaR_;
  //  float fDeltaEta_;
  //float fDeltaPhi_;

  float fEta_;
  float fPhi_;
  float fEtaSC_;
  float fPhiSC_;
  
  float fPt_;
  float fVx_;
  float fVy_;
  float fVz_;
  
  reco::SuperClusterRef refSC;
  bool pivotInBarrel;

     

};

#endif
