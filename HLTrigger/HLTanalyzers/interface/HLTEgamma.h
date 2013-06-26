#ifndef HLTrigger_HLTanalyzers_HLTEgamma_h
#define HLTrigger_HLTanalyzers_HLTEgamma_h


#include <vector>
#include <algorithm>
#include <memory>
#include <map>

#include "TTree.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Common/interface/ValueMap.h"

/** \class HLTEgamma
  *
  * $Date: November 2006
  * $Revision:
  * \author P. Bargassa - Rice U.
  */
class HLTEgamma {
public:
  HLTEgamma();

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  void clear(void);

  /** Analyze the Data */
  void analyze(
      const edm::Handle<reco::GsfElectronCollection>         & electrons,
      const edm::Handle<reco::PhotonCollection>              & photons,
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandle,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackNonIsolMap,
      EcalClusterLazyTools& lazyTools,
      const edm::ESHandle<MagneticField>& theMagField,
      reco::BeamSpot::Point & BSPosition,
      std::vector<edm::Handle<edm::ValueMap<float> > > & eIDValueMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9NonIsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9NonIsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHNonIsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IDIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IDNonIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IDIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IDNonIsoMap,
      const edm::Handle<reco::SuperClusterCollection>        & electronHFClusterHandle, 
      const edm::Handle<reco::RecoEcalCandidateCollection>   & electronHFElectronHandle,  
      const edm::Handle<reco::HFEMClusterShapeAssociationCollection> & electronHFClusterAssociation,  
      const edm::Handle<reco::RecoEcalCandidateCollection>   & activityECAL,   
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityEcalIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityHcalIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityTrackIsoMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityR9Map,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityR9IDMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & activityHoverEHMap,
      TTree* tree);

private:
  struct OpenHLTPhoton;
  struct OpenHLTElectron;

  void MakeL1IsolatedPhotons(
      std::vector<OpenHLTPhoton> & photons,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHIsoMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IDIsoMap,
      EcalClusterLazyTools& lazyTools
      );

  void MakeL1NonIsolatedPhotons(
      std::vector<OpenHLTPhoton> & photons,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9NonIsoMap,  
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHNonIsoMap,  
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonR9IDNonIsoMap,
      EcalClusterLazyTools& lazyTools
      );

  void MakeL1IsolatedElectrons(
      std::vector<OpenHLTElectron> & electrons,
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IsoMap,  
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHIsoMap,  
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalIsolMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IDIsoMap,
      EcalClusterLazyTools& lazyTools,
      const edm::ESHandle<MagneticField>& theMagField,
      reco::BeamSpot::Point & BSPosition  );

  void MakeL1NonIsolatedElectrons(
      std::vector<OpenHLTElectron> & electrons,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9NonIsoMap,   
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & photonHoverEHIsoMap,  
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalIsolMap, 
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & electronR9IDNonIsoMap,
      EcalClusterLazyTools& lazyTools,
      const edm::ESHandle<MagneticField>& theMagField,
      reco::BeamSpot::Point & BSPosition  );

  void CalculateDetaDphi(
		       const edm::ESHandle<MagneticField>& theMagField, 
		       reco::BeamSpot::Point & BSPosition, 
		       const reco::ElectronRef eleref, 
		       float& deltaeta, 
		       float& deltaphi, bool useTrackProjectionToEcal);

  // Tree variables
  float *elpt, *elphi, *eleta, *elet, *ele, *elIP, *elTrkChi2NDF, *elTrkIsoR03, *elECaloIsoR03, *elHCaloIsoR03, *elFbrem; 
  float *eltrkiso, *elecaliso, *elhcaliso; 
  float *elsigmaietaieta, *eldeltaPhiIn,  *eldeltaEtaIn, *elhOverE; 
  float *elscEt, *eld0corr; 
  bool *elqGsfCtfScPixConsistent; 
  int *elmishits; 
  float *eldist, *eldcot;  
  float *photonpt, *photonphi, *photoneta, *photonet, *photone;
  float *photontrkiso, *photonecaliso, *photonhcaliso, *photonhovere, *photonClusShap, *photonr9id;

  float *hecalactivet, *hecalactiveta, *hecalactivphi, *hecalactiveiso, *hecalactivhiso, *hecalactivtiso, *hecalactivhovereh;
  float *hphotet, *hphoteta, *hphotphi, *hphoteiso, *hphothiso, *hphottiso, *hphothovereh;
  float *heleet, *heleeta, *helephi, *helevtxz, *heleE, *helep, *helehiso, *heletiso, *helehovereh, *heleeiso;
  //float *hphotClusShap, *heleClusShap, *heleDeta, *heleDphi;
  //float *hphotR9, *heleR9, *hphotR9ID, *heleR9ID;
  //int *hphotl1iso, *helel1iso, *helePixelSeeds;
  float *hecalactivClusShap,*hphotClusShap, *heleClusShap, *heleDeta, *heleDphi;
  float *hecalactivR9, *hphotR9, *heleR9, *hecalactivR9ID, *hphotR9ID, *heleR9ID;
  int *hecalactivl1iso,*hphotl1iso, *helel1iso, *helePixelSeeds;
  int *eleId, *elNLostHits;//eleId = RL  + 2*RT + 4*L +  4*T  //elNLostHits = conversion rejection  
  bool *elIsEcalDriven;  
  int *heleNewSC;
  int nele, nphoton, nhltecalactiv, nhltgam, nhltele, nhlthfele, nhlthfeclus;
  
  float *hhfelept, *hhfeleeta, *hhfclustere9e25, *hhfcluster2Dcut, *hhfclustereta, *hhfclusterphi; 
  float *hhfclustere1e9, *hhfclustereCOREe9, *hhfclustereSeL;
  
  struct OpenHLTPhoton {
    float Et;
    float eta;
    float phi;
    float ecalIsol;
    float hcalIsol;
    float trackIsol;
    float r9;
    bool  L1Isolated;
    float clusterShape;
    float hovereh;
    float r9ID;
    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };

  struct OpenHLTElectron {
    float Et;
    float eta;
    float phi;
    float E;
    float p;
	 float vtxZ;
    float hcalIsol;
    float trackIsol;
    float ecalIsol;
    bool  L1Isolated;
    int   pixelSeeds;
    bool  newSC;
    float clusterShape;
    float r9;
    float Deta;
    float Dphi;
    float hovereh; 
    float r9ID;
    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };

};

#endif // HLTrigger_HLTanalyzers_HLTEgamma_h
