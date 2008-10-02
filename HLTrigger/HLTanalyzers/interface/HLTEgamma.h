#ifndef HLTrigger_HLTanalyzers_HLTEgamma_h
#define HLTrigger_HLTanalyzers_HLTEgamma_h

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

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

#include "TTree.h"
#include "TFile.h"
#include <vector>
#include <algorithm>
#include <memory>

typedef std::vector<std::string> MyStrings;

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

  /** Analyze the Data */
  void analyze(
      const reco::GsfElectronCollection         * electrons,
      const reco::PhotonCollection              * photons,
      const reco::ElectronCollection            * electronIsoHandle,
      const reco::ElectronCollection            * electronIsoHandleLW,
      const reco::ElectronCollection            * electronNonIsoHandle,
      const reco::ElectronCollection            * electronNonIsoHandleLW,
      const reco::ElectronIsolationMap          * NonIsoTrackEleIsolMap,
      const reco::ElectronIsolationMap          * NonIsoTrackEleIsolMapLW,
      const reco::ElectronIsolationMap          * TrackEleIsolMap,
      const reco::ElectronIsolationMap          * TrackEleIsolMapLW,
      const reco::ElectronPixelSeedCollection   * L1IsoPixelSeedsMap,
      const reco::ElectronPixelSeedCollection   * L1IsoPixelSeedsMapLW,
      const reco::ElectronPixelSeedCollection   * L1NonIsoPixelSeedsMap,
      const reco::ElectronPixelSeedCollection   * L1NonIsoPixelSeedsMapLW,
      const reco::RecoEcalCandidateCollection   * recoIsolecalcands,
      const reco::RecoEcalCandidateCollection   * recoNonIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * EcalIsolMap,
      const reco::RecoEcalCandidateIsolationMap * EcalNonIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalEleIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalEleNonIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalNonIsolMap,
      const reco::RecoEcalCandidateIsolationMap * TrackIsolMap,
      const reco::RecoEcalCandidateIsolationMap * TrackNonIsolMap,
      TTree* tree);

private:

  void MakeL1IsolatedPhotons(
      const reco::RecoEcalCandidateCollection   * recoIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * EcalIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalIsolMap,
      const reco::RecoEcalCandidateIsolationMap * TrackIsolMap);

  void MakeL1NonIsolatedPhotons(
      const reco::RecoEcalCandidateCollection   * recoNonIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * EcalNonIsolMap,
      const reco::RecoEcalCandidateIsolationMap * HcalNonIsolMap,
      const reco::RecoEcalCandidateIsolationMap * TrackNonIsolMap);

  void MakeL1IsolatedElectrons(
      const reco::ElectronCollection            * electronIsoHandle,
      const reco::RecoEcalCandidateCollection   * recoIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * HcalEleIsolMap,
      const reco::ElectronPixelSeedCollection   * L1IsoPixelSeedsMap,
      const reco::ElectronIsolationMap          * TrackEleIsolMap);

  void MakeL1NonIsolatedElectrons(
      const reco::ElectronCollection            * electronNonIsoHandle,
      const reco::RecoEcalCandidateCollection   * recoNonIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * HcalEleIsolMap,
      const reco::ElectronPixelSeedCollection   * L1NonIsoPixelSeedsMap,
      const reco::ElectronIsolationMap          * TrackEleIsolMap);

  void MakeL1IsolatedElectronsLargeWindows(
      const reco::ElectronCollection            * electronIsoHandle,
      const reco::RecoEcalCandidateCollection   * recoIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * HcalEleIsolMap,
      const reco::ElectronPixelSeedCollection   * L1IsoPixelSeedsMap,
      const reco::ElectronIsolationMap          * TrackEleIsolMap);

  void MakeL1NonIsolatedElectronsLargeWindows(
      const reco::ElectronCollection            * electronNonIsoHandle,
      const reco::RecoEcalCandidateCollection   * recoNonIsolecalcands,
      const reco::RecoEcalCandidateIsolationMap * HcalEleIsolMap,
      const reco::ElectronPixelSeedCollection   * L1NonIsoPixelSeedsMap,
      const reco::ElectronIsolationMap          * TrackEleIsolMap);

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
