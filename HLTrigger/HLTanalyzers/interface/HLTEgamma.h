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

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEgamma/Examples/plugins/ElectronIDAnalyzer.h"
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
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandleLW,
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandleSS,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandleLW,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandleSS,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMap,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMapLW,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMapSS,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMapLW,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMapSS,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMapLW,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMapSS,
      const edm::Handle<reco::ElectronSeedCollection>        & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1NonIsoPixelSeedsMapLW,
      const edm::Handle<reco::ElectronSeedCollection>        & L1NonIsoPixelSeedsMapSS,
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
      EcalClusterLazyTools& lazyTools );

  void MakeL1NonIsolatedPhotons(
      std::vector<OpenHLTPhoton> & photons,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackNonIsolMap,
      EcalClusterLazyTools& lazyTools );

  void MakeL1IsolatedElectrons(
      std::vector<OpenHLTElectron> & electrons,
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronSeedCollection>        & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap,
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
      EcalClusterLazyTools& lazyTools,
      const edm::ESHandle<MagneticField>& theMagField,
      reco::BeamSpot::Point & BSPosition  );

void CalculateDetaDphi(
		       const edm::ESHandle<MagneticField>& theMagField, 
		       reco::BeamSpot::Point & BSPosition, 
		       const reco::ElectronRef eleref, 
		       float& deltaeta, 
		       float& deltaphi );

  // Tree variables
  float *elpt, *elphi, *eleta, *elet, *ele;
  float *photonpt, *photonphi, *photoneta, *photonet, *photone;
  float *hphotet, *hphoteta, *hphotphi, *hphoteiso, *hphothiso, *hphottiso;
  float *heleet, *heleeta, *helephi, *heleE, *helep, *helehiso, *heletiso;
  float *heleetLW, *heleetaLW, *helephiLW, *heleELW, *helepLW, *helehisoLW, *heletisoLW;
  float *heleetSS, *heleetaSS, *helephiSS, *heleESS, *helepSS, *helehisoSS, *heletisoSS;
  float *hphotClusShap, *heleClusShap, *heleDeta, *heleDphi, *heleClusShapLW, *heleDetaLW, *heleDphiLW, *heleClusShapSS, *heleDetaSS, *heleDphiSS;
  int *hphotl1iso, *helel1iso, *helePixelSeeds, *helel1isoLW, *helePixelSeedsLW, *helel1isoSS, *helePixelSeedsSS;
  int *eleId;// RL  + 2*RT + 4*L +  4*T 
  int *heleNewSC, *heleNewSCLW, *heleNewSCSS;
  int nele, nphoton, nhltgam, nhltele, nhlteleLW, nhlteleSS;

  struct OpenHLTPhoton {
    float Et;
    float eta;
    float phi;
    float ecalIsol;
    float hcalIsol;
    float trackIsol;
    bool  L1Isolated;
    float clusterShape;
    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };

  struct OpenHLTElectron {
    float Et;
    float eta;
    float phi;
    float E;
    float p;
    float hcalIsol;
    float trackIsol;
    bool  L1Isolated;
    int   pixelSeeds;
    bool  newSC;
    float clusterShape;
    float Deta;
    float Dphi;
    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };

};

#endif // HLTrigger_HLTanalyzers_HLTEgamma_h
