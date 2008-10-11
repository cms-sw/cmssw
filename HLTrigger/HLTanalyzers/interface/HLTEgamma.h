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
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

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
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandleLW,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMap,
      const edm::Handle<reco::ElectronIsolationMap>          & NonIsoTrackEleIsolMapLW,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMapLW,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1IsoPixelSeedsMapLW,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1NonIsoPixelSeedsMapLW,
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
      TTree* tree);

private:

  void MakeL1IsolatedPhotons(
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackIsolMap);

  void MakeL1NonIsolatedPhotons(
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & EcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalNonIsolMap,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & TrackNonIsolMap);

  void MakeL1IsolatedElectrons(
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap);

  void MakeL1NonIsolatedElectrons(
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap);

  void MakeL1IsolatedElectronsLargeWindows(
      const edm::Handle<reco::ElectronCollection>            & electronIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1IsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap);

  void MakeL1NonIsolatedElectronsLargeWindows(
      const edm::Handle<reco::ElectronCollection>            & electronNonIsoHandle,
      const edm::Handle<reco::RecoEcalCandidateCollection>   & recoNonIsolecalcands,
      const edm::Handle<reco::RecoEcalCandidateIsolationMap> & HcalEleIsolMap,
      const edm::Handle<reco::ElectronPixelSeedCollection>   & L1NonIsoPixelSeedsMap,
      const edm::Handle<reco::ElectronIsolationMap>          & TrackEleIsolMap);

  // Tree variables
  float *elpt, *elphi, *eleta, *elet, *ele;
  float *photonpt, *photonphi, *photoneta, *photonet, *photone;
  float *hphotet, *hphoteta, *hphotphi, *hphoteiso, *hphothiso, *hphottiso;
  float *heleet, *heleeta, *helephi, *heleE, *helep, *helehiso, *heletiso;
  float *heleetLW, *heleetaLW, *helephiLW, *heleELW, *helepLW, *helehisoLW, *heletisoLW;
  int *hphotl1iso, *helel1iso, *helePixelSeeds, *helel1isoLW, *helePixelSeedsLW;
  int *heleNewSC, *heleNewSCLW;
  int nele, nphoton, nhltgam, nhltele, nhlteleLW;

  class myHLTPhoton {
  public:
    float Et;
    float eta;
    float phi;
    float ecalIsol;
    float hcalIsol;
    float trackIsol;
    bool  L1Isolated;

    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };
  std::vector<myHLTPhoton> theHLTPhotons;

  class myHLTElectron {
  public:
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

    float et() const { return Et; } // Function defined as such to be compatible with EtGreater()
  };
  std::vector<myHLTElectron> theHLTElectrons;
  std::vector<myHLTElectron> theHLTElectronsLargeWindows;

};

#endif // HLTrigger_HLTanalyzers_HLTEgamma_h
