/* -*- C++ -*-

Package: L1CaloTrigger
Class: l1tNNCaloTauProducer
Frinedly name: The TauMinator

\class l1tNNCaloTauProducer l1tNNCaloTauProducer.cc

Description: 
Perform reconstruction and identification of tau 
candidates at L1 Trigger with a CNN.

Implementation:
Take as input the HCAL TPs, the ECAL TPs from
l1tEGammaClusterEmuProducer, and the HGCAL TPs
from l1tHGCalTowerProducer and l1tHGCalBackEndLayer2Producer.
Proceed to clustering of trigger towers in NxM
clusters, match to HGcal 3D clusters in the endcap.
Finally apply the CNNs.

Original Author: Jona Motta
Created: Tue May 30th 2023

*/

#include <iostream>
#include <vector>
#include <cmath>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class l1tNNCaloTauProducer : public edm::stream::EDProducer<> {
public:
  explicit l1tNNCaloTauProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  //----edm control---
  void produce(edm::Event&, const edm::EventSetup&) override;

  //----private functions----
  int tower_dIPhi(int& iPhi_1, int& iPhi_2) const;
  int tower_dIEta(int& iEta_1, int& iEta_2) const;
  int endcap_iphi(float& phi) const;
  int endcap_ieta(float& eta) const;
  float inputQuantizer(float inputF, float LSB, int nbits);
  float inputScaler(float inputF, std::string feature);

  //----tokens and handles----
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowersToken;
  edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

  edm::EDGetToken hgcalTowersToken;
  edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

  edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> HGClusterToken;
  edm::Handle<l1t::HGCalMulticlusterBxCollection> HGClusterHandle;

  //----private variables----
  enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };
  UseEmInterp scenario;
  StringCutObjectSelector<l1t::HGCalMulticluster> preEmId;
  l1tpf::HGC3DClusterEgID VsPuId;

  double EcalEtMinForClustering;
  double HcalEtMinForClustering;
  double EtMinForSeeding;
  double EtaRestriction;
  double CB_CE_split;

  std::string CNNmodel_CB_path;
  std::string DNNident_CB_path;
  std::string DNNcalib_CB_path;

  std::string CNNmodel_CE_path;
  std::string DNNident_CE_path;
  std::string DNNcalib_CE_path;
  std::string FeatScaler_CE_path;
  boost::property_tree::ptree FeatScaler_CE;

  tensorflow::GraphDef* CNNmodel_CB;
  tensorflow::GraphDef* DNNident_CB;
  tensorflow::GraphDef* DNNcalib_CB;

  tensorflow::Session* CNNmodel_CBsession;
  tensorflow::Session* DNNident_CBsession;
  tensorflow::Session* DNNcalib_CBsession;

  tensorflow::GraphDef* CNNmodel_CE;
  tensorflow::GraphDef* DNNident_CE;
  tensorflow::GraphDef* DNNcalib_CE;

  tensorflow::Session* CNNmodel_CEsession;
  tensorflow::Session* DNNident_CEsession;
  tensorflow::Session* DNNcalib_CEsession;

  double IdWp90_CB;
  double IdWp95_CB;
  double IdWp99_CB;

  double IdWp90_CE;
  double IdWp95_CE;
  double IdWp99_CE;

  // hardoced dimensions of the tower clusters
  const int seedIdx = 22;
  const int IEta_dim = 5;
  const int IPhi_dim = 9;
  const float Eta_dim = 0.2;
  const float Phi_dim = 0.4;
  const float Eta_dim_seed = 0.35;
  const float Phi_dim_seed = 0.7;
  const float Eta_limit = 2.83;

  // classes of objects used only in this producer
  class SimpleTowerHit {
  public:
    float towerEta = -99.;
    float towerPhi = -99.;
    float towerEm = 0.;
    float towerHad = 0.;
    float l1egTowerEt = 0.;
    float towerEt = 0.;
    int towerIeta = -99;
    int towerIphi = -99;
    bool isBarrel = true;
    bool stale = false;
    bool stale4seed = false;
  };

  class SimpleTowerCluster {
  public:
    bool barrelSeeded = false;
    int seedIeta = -99;
    int seedIphi = -99;
    float seedEta = -99.;
    float seedPhi = -99.;
    float rawEt = 0.;
    float IDscore = -99.;
    float calibPt = -99.;

    std::vector<SimpleTowerHit> towerHits;

    void InitHits(int N, int M) { towerHits.resize(N * M); }
  };

  class SimpleHGCluster {
  public:
    float pt = -99.;
    float eta = -99.;
    float phi = -99.;
    float showerlength = -99.;
    float coreshowerlength = -99.;
    float spptot = -99.;
    float szz = -99.;
    float srrtot = -99.;
    float meanz = -99.;
    bool stale = false;
  };
};

/*
████████ ██   ██ ██████     ████████  █████  ██   ██ ███    ███ ██ ███    ██  █████  ████████  ██████  ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ████  ████ ██ ████   ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ███████ █████         ██    ███████ ██   ██ ██ ████ ██ ██ ██ ██  ██ ███████    ██    ██    ██ ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ██  ██  ██ ██ ██  ██ ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ██   ██ ██████        ██    ██   ██ ███████ ██      ██ ██ ██   ████ ██   ██    ██     ██████  ██    ██
*/

// ----Constructor and Destructor -----
l1tNNCaloTauProducer::l1tNNCaloTauProducer(const edm::ParameterSet& iConfig)
    : l1TowersToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),

      HGClusterToken(
          consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      scenario(UseEmInterp::No),
      preEmId(iConfig.getParameter<std::string>("preEmId")),
      VsPuId(iConfig.getParameter<edm::ParameterSet>("VsPuId")),

      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      EtaRestriction(iConfig.getParameter<double>("EtaRestriction")),
      CB_CE_split(iConfig.getParameter<double>("CB_CE_split")),

      CNNmodel_CB_path(iConfig.getParameter<std::string>("CNNmodel_CB_path")),
      DNNident_CB_path(iConfig.getParameter<std::string>("DNNident_CB_path")),
      DNNcalib_CB_path(iConfig.getParameter<std::string>("DNNcalib_CB_path")),
      CNNmodel_CE_path(iConfig.getParameter<std::string>("CNNmodel_CE_path")),
      DNNident_CE_path(iConfig.getParameter<std::string>("DNNident_CE_path")),
      DNNcalib_CE_path(iConfig.getParameter<std::string>("DNNcalib_CE_path")),
      FeatScaler_CE_path(iConfig.getParameter<std::string>("FeatScaler_CE_path")),

      IdWp90_CB(iConfig.getParameter<double>("IdWp90_CB")),
      IdWp95_CB(iConfig.getParameter<double>("IdWp95_CB")),
      IdWp99_CB(iConfig.getParameter<double>("IdWp99_CB")),

      IdWp90_CE(iConfig.getParameter<double>("IdWp90_CE")),
      IdWp95_CE(iConfig.getParameter<double>("IdWp95_CE")),
      IdWp99_CE(iConfig.getParameter<double>("IdWp99_CE")) {
  // Create sessions for Tensorflow inferece
  CNNmodel_CB = tensorflow::loadGraphDef(edm::FileInPath(CNNmodel_CB_path).fullPath());
  CNNmodel_CBsession = tensorflow::createSession(CNNmodel_CB);

  DNNident_CB = tensorflow::loadGraphDef(edm::FileInPath(DNNident_CB_path).fullPath());
  DNNident_CBsession = tensorflow::createSession(DNNident_CB);

  DNNcalib_CB = tensorflow::loadGraphDef(edm::FileInPath(DNNcalib_CB_path).fullPath());
  DNNcalib_CBsession = tensorflow::createSession(DNNcalib_CB);

  CNNmodel_CE = tensorflow::loadGraphDef(edm::FileInPath(CNNmodel_CE_path).fullPath());
  CNNmodel_CEsession = tensorflow::createSession(CNNmodel_CE);

  DNNident_CE = tensorflow::loadGraphDef(edm::FileInPath(DNNident_CE_path).fullPath());
  DNNident_CEsession = tensorflow::createSession(DNNident_CE);

  DNNcalib_CE = tensorflow::loadGraphDef(edm::FileInPath(DNNcalib_CE_path).fullPath());
  DNNcalib_CEsession = tensorflow::createSession(DNNcalib_CE);

  // Read features scaler
  boost::property_tree::read_json(edm::FileInPath(FeatScaler_CE_path).fullPath(), FeatScaler_CE);

  // Initialize HGCAL BDTs
  if (!VsPuId.method().empty()) {
    VsPuId.prepareTMVA();
  }

  // Create produced outputs
  produces<BXVector<l1t::Tau>>("L1NNCaloTauCollectionBXV");

  // Settings output
  edm::LogInfo("Settings") << "EtaRestriction = " << EtaRestriction << " , CB_CE_split = " << CB_CE_split
                           << " , EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering
                           << " , EcalTpEtMin = " << EcalEtMinForClustering << std::endl;
}

void l1tNNCaloTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& eSetup) {
  // Output collection
  std::unique_ptr<BXVector<l1t::Tau>> L1NNCaloTauCollectionBXV(new l1t::TauBxCollection);

  // Create and Fill collection of all calotowers and their attributes
  std::vector<SimpleTowerHit> l1CaloTowers;

  iEvent.getByToken(l1TowersToken, l1CaloTowerHandle);
  int warnings = 0;
  for (auto& hit : *l1CaloTowerHandle.product()) {
    // Skip this weird towers and store warning
    if (hit.towerIEta() == -1016 && hit.towerIPhi() == -962) {
      warnings += 1;
      continue;
    }

    SimpleTowerHit l1Hit;
    l1Hit.isBarrel = true;
    l1Hit.l1egTowerEt = hit.l1egTowerEt();
    l1Hit.towerEta = hit.towerEta();
    l1Hit.towerPhi = hit.towerPhi();
    l1Hit.towerEm = hit.ecalTowerEt();
    l1Hit.towerHad = hit.hcalTowerEt();
    l1Hit.towerEt = l1Hit.towerEm + l1Hit.towerHad + l1Hit.l1egTowerEt;
    l1Hit.towerIeta = hit.towerIEta();
    l1Hit.towerIphi = hit.towerIPhi();

    l1CaloTowers.push_back(l1Hit);
  }
  if (warnings != 0) {
    edm::LogWarning("BrokenTowers") << " ** WARNING : FOUND " << warnings << " TOWERS WITH towerIeta=-1016 AND towerIphi=-962" << std::endl;
  }

  iEvent.getByToken(hgcalTowersToken, hgcalTowersHandle);
  for (auto& hit : *hgcalTowersHandle.product()) {
    SimpleTowerHit l1Hit;
    l1Hit.isBarrel = false;
    l1Hit.l1egTowerEt = 0.0;
    l1Hit.towerEta = hit.eta();
    l1Hit.towerPhi = hit.phi();
    l1Hit.towerEm = hit.etEm();
    l1Hit.towerHad = hit.etHad();
    l1Hit.towerEt = l1Hit.towerEm + l1Hit.towerHad;
    l1Hit.towerIeta = endcap_ieta(l1Hit.towerEta);  // computed and filled but not used
    l1Hit.towerIphi = endcap_iphi(l1Hit.towerPhi);  // computed and filled but not used

    l1CaloTowers.push_back(l1Hit);
  }

  // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
  std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const SimpleTowerHit& a, SimpleTowerHit& b) {
    return a.towerEt > b.towerEt;
  });

  // Create and Fill the collection of 3D clusters and their attributes
  std::vector<SimpleHGCluster> AllHGClusters;
  iEvent.getByToken(HGClusterToken, HGClusterHandle);

  for (auto cl3dIt = HGClusterHandle->begin(0); cl3dIt != HGClusterHandle->end(0); ++cl3dIt) {
    auto& cl3d = *cl3dIt;

    // Implement cl3d PU ID as done in
    // https://github.com/cms-sw/cmssw/blob/master/L1Trigger/Phase2L1ParticleFlow/plugins/PFClusterProducerFromHGC3DClusters.cc#L120
    bool isEM = preEmId(*cl3dIt);
    l1t::PFCluster cluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), cl3d.hOverE());
    if (scenario == UseEmInterp::EmOnly)  // for emID objs, use EM interp as pT and set H = 0
    {
      if (isEM) {
        float pt_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
        float hoe_new = 0.;
        cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
      }
    } else if (scenario == UseEmInterp::AllKeepHad)  // for all objs, replace EM part with EM interp, preserve H
    {
      float had_old = cl3d.pt() - cluster.emEt();
      float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float pt_new = had_old + em_new;
      float hoe_new = em_new > 0 ? (had_old / em_new) : -1;
      cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
    } else if (scenario == UseEmInterp::AllKeepTot)  // for all objs, replace EM part with EM interp, preserve pT
    {
      float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float hoe_new = em_new > 0 ? (cl3d.pt() / em_new - 1) : -1;
      cluster = l1t::PFCluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), hoe_new, isEM);
    }

    if (!VsPuId.method().empty()) {
      int id = VsPuId.passID(*cl3dIt, cluster);
      if (!id) {
        continue;
      }  // skip cl3d if it does not pass puid
    }

    SimpleHGCluster HGCluster;
    HGCluster.pt = cl3d.pt();
    HGCluster.eta = cl3d.eta();
    HGCluster.phi = cl3d.phi();
    HGCluster.showerlength = cl3d.showerLength();
    HGCluster.coreshowerlength = cl3d.coreShowerLength();
    HGCluster.spptot = cl3d.sigmaPhiPhiTot();
    HGCluster.szz = cl3d.sigmaZZ();
    HGCluster.srrtot = cl3d.sigmaRRTot();
    HGCluster.meanz = cl3d.zBarycenter();

    AllHGClusters.push_back(HGCluster);
  }

  // Order the collection in pt (the input to the GCT will be pt ordered)
  std::sort(begin(AllHGClusters), end(AllHGClusters), [](const SimpleHGCluster& a, SimpleHGCluster& b) {
    return a.pt > b.pt;
  });

  // Make NxM TowerClusters and HGClusters collections for TauMinator
  std::vector<SimpleTowerCluster> l1TowerClustersNxM_CB;
  std::vector<SimpleTowerCluster> l1TowerClustersNxM_CE;
  std::vector<SimpleHGCluster> HGClusters;

  // Supporting collection of endcap clusters before cl3d matching
  std::vector<SimpleTowerCluster> AllL1TowerClustersNxM_CE;

  bool caloTauSeedingFinished = false;
  // Loop for seeding of clNxM objects
  while (!caloTauSeedingFinished) {
    SimpleTowerCluster clNxM;
    clNxM.InitHits(IEta_dim, IPhi_dim);
    bool seeded = false;

    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip seeding in towers that would make the cluster extend in HF
      // Skip l1CaloTowers which are already used by this clusters' mask
      if (abs(l1CaloTower.towerEta) > Eta_limit || abs(l1CaloTower.towerEta) > EtaRestriction ||
          l1CaloTower.stale4seed) {
        continue;
      }

      // If not seded do the seeding
      if (!seeded) {
        // The leading unused tower has ET < min, stop jet clustering
        if (l1CaloTower.towerEt < EtMinForSeeding) {
          caloTauSeedingFinished = true;
          continue;
        }

        clNxM.seedIeta = l1CaloTower.towerIeta;
        clNxM.seedIphi = l1CaloTower.towerIphi;
        clNxM.seedEta = l1CaloTower.towerEta;
        clNxM.seedPhi = l1CaloTower.towerPhi;
        if (l1CaloTower.isBarrel) {
          clNxM.barrelSeeded = true;
        }

        clNxM.rawEt += l1CaloTower.towerEt;
        clNxM.towerHits[seedIdx] = l1CaloTower;
        l1CaloTower.stale4seed = true;
        l1CaloTower.stale = true;
        seeded = true;

        continue;
      }

      int d_iEta = 99;
      int d_iPhi = 99;
      float d_Eta = 99.;
      float d_Phi = 99.;
      // Ese iEta/iPhi comparisons in the barrel and eta/phi in HGCal
      if (clNxM.barrelSeeded && l1CaloTower.isBarrel) {
        d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
        d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);
      } else {
        d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
        d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);
      }

      // Stale tower for seeding if it would lead to overalp between clusters
      if ((abs(d_iEta) <= IEta_dim - 1 && abs(d_iPhi) <= IPhi_dim - 1) ||
          (abs(d_Eta) < Eta_dim_seed && abs(d_Phi) < Phi_dim_seed)) {
        l1CaloTower.stale4seed = true;
      }

    }  // End for loop over TPs

    // Pushback seeds split in barrel and endcap
    if (seeded) {
      if (abs(clNxM.seedEta) < CB_CE_split) {
        l1TowerClustersNxM_CB.push_back(clNxM);
      } else {
        AllL1TowerClustersNxM_CE.push_back(clNxM);
      }
    }

  }  // End while loop of TowerClusters seeding

  // Loop for barrel NxM TowerClusters clustering starting from the seeds
  for (auto& clNxM : l1TowerClustersNxM_CB) {
    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      int d_iEta = 99;
      int d_iPhi = 99;
      float d_Eta = 99.;
      float d_Phi = 99.;
      int hitIdx = 99.;
      // Use iEta/iPhi comparisons in the barrel and use eta/phi in HGCal
      if (l1CaloTower.isBarrel) {
        d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
        d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);

        hitIdx = d_iEta * IPhi_dim + d_iPhi + seedIdx;
      } else {
        d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
        d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);

        int dieta = d_Eta / 0.0807;  // minimal difference in endcap is 0.0808
        int diphi = d_Phi / 0.0872;
        hitIdx = dieta * IPhi_dim + diphi + seedIdx;
      }

      // Cluster all towers in a NxM towers mask
      if ((abs(d_iEta) <= (IEta_dim - 1) / 2 && abs(d_iPhi) <= (IPhi_dim - 1) / 2) ||
          (abs(d_Eta) < Eta_dim && abs(d_Phi) < Phi_dim)) {
        clNxM.rawEt += l1CaloTower.towerEt;
        clNxM.towerHits[hitIdx] = l1CaloTower;
        l1CaloTower.stale = true;
      }

    }  // End for loop over TPs

  }  // End while loop of barrel TowerClusters creation

  // In the endcap cross-loop over clNxM and cl3d to match them
  // (we can do it before full clustering just using the seed info)
  for (auto& clNxM : AllL1TowerClustersNxM_CE) {
    bool matched = false;
    for (auto& HGCluster : AllHGClusters) {
      // In case the clNxM or HGCluster have already been matched just continue through the list to the end
      // only use cl3ds above 4GeV
      if (matched || HGCluster.stale || HGCluster.pt < 4) {
        continue;
      }

      float d_Eta = HGCluster.eta - clNxM.seedEta;
      float d_Phi = reco::deltaPhi(HGCluster.phi, clNxM.seedPhi);
      float d_R2 = pow(d_Eta, 2) + pow(d_Phi, 2);

      if (d_R2 < 0.25) {
        HGCluster.stale = true;
        HGClusters.push_back(HGCluster);
        l1TowerClustersNxM_CE.push_back(clNxM);
        matched = true;
      }

    }  // End for loop over cl3ds

  }  // End for loop over clNxM

  // Loop for endcap matched NxM TowerClusters clustering starting from the seeds just found
  for (auto& clNxM : l1TowerClustersNxM_CE) {
    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      int d_iEta = 99;
      int d_iPhi = 99;
      float d_Eta = 99.;
      float d_Phi = 99.;
      int hitIdx = 99.;
      // Use iEta/iPhi comparisons in the endcap and use eta/phi in HGCal
      if (l1CaloTower.isBarrel) {
        d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
        d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);

        hitIdx = d_iEta * IPhi_dim + d_iPhi + seedIdx;
      } else {
        d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
        d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);

        int dieta = d_Eta / 0.0807;  // minimal difference in endcap is 0.0808
        int diphi = d_Phi / 0.0872;
        hitIdx = dieta * IPhi_dim + diphi + seedIdx;
      }

      // Cluster all towers in a NxM towers mask
      if ((abs(d_iEta) <= (IEta_dim - 1) / 2 && abs(d_iPhi) <= (IPhi_dim - 1) / 2) ||
          (abs(d_Eta) < Eta_dim && abs(d_Phi) < Phi_dim)) {
        clNxM.rawEt += l1CaloTower.towerEt;
        clNxM.towerHits[hitIdx] = l1CaloTower;
        l1CaloTower.stale = true;
      }

    }  // End for loop over TPs

  }  // End while loop of endcap TowerClusters creation

  // Barrel TauMinator application
  tensorflow::setLogging("2");
  int batchSize_CB = (int)(l1TowerClustersNxM_CB.size());
  tensorflow::TensorShape imageShape_CB({batchSize_CB, IEta_dim, IPhi_dim, 2});
  tensorflow::TensorShape positionShape_CB({batchSize_CB, 2});
  tensorflow::Tensor TowerClusterImage_CB(tensorflow::DT_FLOAT, imageShape_CB);
  tensorflow::Tensor TowerClusterPosition_CB(tensorflow::DT_FLOAT, positionShape_CB);

  int clIdx = 0;
  for (auto& clNxM : l1TowerClustersNxM_CB) {
    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CB.tensor<float, 4>()(clIdx, eta, phi, 0) =
            inputQuantizer(clNxM.towerHits[towerIdx].l1egTowerEt + clNxM.towerHits[towerIdx].towerEm, 0.25, 10);
        TowerClusterImage_CB.tensor<float, 4>()(clIdx, eta, phi, 1) =
            inputQuantizer(clNxM.towerHits[towerIdx].towerHad, 0.25, 10);
      }
    }

    TowerClusterPosition_CB.tensor<float, 2>()(clIdx, 0) = clNxM.seedEta;
    TowerClusterPosition_CB.tensor<float, 2>()(clIdx, 1) = clNxM.seedPhi;

    clIdx++;  // Increase batch index
  }

  // Apply CNN model
  tensorflow::NamedTensorList CNNmodel_CBinputList = {{"TowerClusterImage", TowerClusterImage_CB},
                                                      {"TowerClusterPosition", TowerClusterPosition_CB}};
  std::vector<tensorflow::Tensor> CNNmodel_CBoutputs;
  tensorflow::run(
      CNNmodel_CBsession, CNNmodel_CBinputList, {"TauMinator_CB_conv/middleMan/concat"}, &CNNmodel_CBoutputs);
  tensorflow::NamedTensorList DNN_CBinputsList = {{"middleMan", CNNmodel_CBoutputs[0]}};

  // Apply DNN for identification
  std::vector<tensorflow::Tensor> DNN_CBoutputsIdent;
  tensorflow::run(
      DNNident_CBsession, DNN_CBinputsList, {"TauMinator_CB_ident/sigmoid_IDout/Sigmoid"}, &DNN_CBoutputsIdent);

  // Apply DNN for calibration
  std::vector<tensorflow::Tensor> DNN_CBoutputsCalib;
  tensorflow::run(DNNcalib_CBsession, DNN_CBinputsList, {"TauMinator_CB_calib/LIN_DNNout/Relu"}, &DNN_CBoutputsCalib);

  // Fill TauMinator output variables of TowerClusters
  clIdx = 0;
  for (auto& clNxM : l1TowerClustersNxM_CB) {
    clNxM.IDscore = DNN_CBoutputsIdent[0].matrix<float>()(0, clIdx);
    clNxM.calibPt = DNN_CBoutputsCalib[0].matrix<float>()(0, clIdx);
    clIdx++;  // Increase batch index
  }

  // Endcap TauMinator application
  int batchSize_CE = (int)(l1TowerClustersNxM_CE.size());
  tensorflow::TensorShape imageShape_CE({batchSize_CE, IEta_dim, IPhi_dim, 2});
  tensorflow::TensorShape positionShape_CE({batchSize_CE, 2});
  tensorflow::TensorShape cl3dfeatShape_CE({batchSize_CE, 8});
  tensorflow::Tensor TowerClusterImage_CE(tensorflow::DT_FLOAT, imageShape_CE);
  tensorflow::Tensor TowerClusterPosition_CE(tensorflow::DT_FLOAT, positionShape_CE);
  tensorflow::Tensor Cl3dShapeFeatures_CE(tensorflow::DT_FLOAT, cl3dfeatShape_CE);

  clIdx = 0;
  for (auto& clNxM : l1TowerClustersNxM_CE) {
    // Indexing of cl3ds is the same as the one of clNxMs
    SimpleHGCluster HGClu = HGClusters[clIdx];

    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CE.tensor<float, 4>()(clIdx, eta, phi, 0) =
            inputQuantizer(clNxM.towerHits[towerIdx].l1egTowerEt + clNxM.towerHits[towerIdx].towerEm, 0.25, 10);
        TowerClusterImage_CE.tensor<float, 4>()(clIdx, eta, phi, 1) =
            inputQuantizer(clNxM.towerHits[towerIdx].towerHad, 0.25, 10);
      }
    }

    TowerClusterPosition_CE.tensor<float, 2>()(clIdx, 0) = clNxM.seedEta;
    TowerClusterPosition_CE.tensor<float, 2>()(clIdx, 1) = clNxM.seedPhi;

    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 0) = inputScaler(inputQuantizer(HGClu.pt, 0.25, 14), "pt");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 1) =
        inputScaler(inputQuantizer(abs(HGClu.eta) - 1.321, 0.004, 9), "eta");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 2) = inputScaler(HGClu.showerlength, "showerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 3) = inputScaler(HGClu.coreshowerlength, "coreshowerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 4) =
        inputScaler(inputQuantizer(HGClu.spptot, 0.0000153, 16), "spptot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 5) = inputScaler(inputQuantizer(HGClu.szz, 0.00153, 16), "szz");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 6) =
        inputScaler(inputQuantizer(HGClu.srrtot, 0.0000153, 16), "srrtot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 7) =
        inputScaler(inputQuantizer(10 * (abs(HGClu.meanz) - 321.05), 0.5, 12), "meanz");

    clIdx++;  // Increase batch index
  }

  // Apply CNN model
  tensorflow::NamedTensorList CNNmodel_CEinputList = {{"TowerClusterImage", TowerClusterImage_CE},
                                                      {"TowerClusterPosition", TowerClusterPosition_CE},
                                                      {"AssociatedCl3dFeatures", Cl3dShapeFeatures_CE}};
  std::vector<tensorflow::Tensor> CNNmodel_CEoutputs;
  tensorflow::run(
      CNNmodel_CEsession, CNNmodel_CEinputList, {"TauMinator_CE_conv/middleMan/concat"}, &CNNmodel_CEoutputs);
  tensorflow::NamedTensorList DNN_CEinputsList = {{"middleMan", CNNmodel_CEoutputs[0]}};

  // Apply DNN for identification
  std::vector<tensorflow::Tensor> DNN_CEoutputsIdent;
  tensorflow::run(
      DNNident_CEsession, DNN_CEinputsList, {"TauMinator_CE_ident/sigmoid_IDout/Sigmoid"}, &DNN_CEoutputsIdent);

  // Apply DNN for calibration
  std::vector<tensorflow::Tensor> DNN_CEoutputsCalib;
  tensorflow::run(DNNcalib_CEsession, DNN_CEinputsList, {"TauMinator_CE_calib/LIN_DNNout/Relu"}, &DNN_CEoutputsCalib);

  // Fill TauMinator output variables of TowerClusters
  clIdx = 0;
  for (auto& clNxM : l1TowerClustersNxM_CE) {
    clNxM.IDscore = DNN_CEoutputsIdent[0].matrix<float>()(0, clIdx);
    clNxM.calibPt = DNN_CEoutputsCalib[0].matrix<float>()(0, clIdx);
    clIdx++;  // Increase batch index
  }

  // Fill the output collection of L1 taus
  for (auto& clNxM : l1TowerClustersNxM_CB) {
    // Apply eta restriction
    if (abs(clNxM.seedEta) > EtaRestriction) {
      continue;
    }

    // Assign increasing quality to higher scoring candidates
    int quality = 0;
    // 99% WP
    if (clNxM.IDscore > IdWp99_CB) {
      quality = 1;
    }
    // 95% WP
    if (clNxM.IDscore > IdWp95_CB) {
      quality = 2;
    }
    // 90% WP
    if (clNxM.IDscore > IdWp90_CB) {
      quality = 3;
    }

    reco::Candidate::PolarLorentzVector tauP4 =
        reco::Candidate::PolarLorentzVector(clNxM.calibPt, clNxM.seedEta, clNxM.seedPhi, 0);

    // store ID score multiplied by 10E4 to have good precision even using the Phase1 tau int iso format
    // (this is stored just in case for possible additional offline studies)
    // tau initialisation =  (p4,    pt,            eta,           phi,           qual,    iso)
    l1t::Tau l1Tau = l1t::Tau(tauP4, clNxM.calibPt, clNxM.seedEta, clNxM.seedPhi, quality, clNxM.IDscore * 10E4);
    l1Tau.setTowerIEta(clNxM.seedIeta);
    l1Tau.setTowerIPhi(clNxM.seedIphi);
    l1Tau.setRawEt(clNxM.rawEt);

    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }

  for (auto& clNxM : l1TowerClustersNxM_CE) {
    // Apply eta restriction
    if (abs(clNxM.seedEta) > EtaRestriction) {
      continue;
    }

    // Assign increasing quality to higher scoring candidates
    int quality = 0;
    // 99% WP
    if (clNxM.IDscore > IdWp99_CE) {
      quality = 1;
    }
    // 95% WP
    if (clNxM.IDscore > IdWp95_CE) {
      quality = 2;
    }
    // 90% WP
    if (clNxM.IDscore > IdWp90_CE) {
      quality = 3;
    }

    reco::Candidate::PolarLorentzVector tauP4 =
        reco::Candidate::PolarLorentzVector(clNxM.calibPt, clNxM.seedEta, clNxM.seedPhi, 0);

    // store ID score multiplied by 10E4 to have good precision even using the Phase1 tau int iso format
    // (this is stored just in case for possible additional offline studies)
    // tau initialisation =  (p4,    pt,            eta,           phi,           qual,    iso)
    l1t::Tau l1Tau = l1t::Tau(tauP4, clNxM.calibPt, clNxM.seedEta, clNxM.seedPhi, quality, clNxM.IDscore * 10E4);
    l1Tau.setTowerIEta(clNxM.seedIeta);
    l1Tau.setTowerIPhi(clNxM.seedIphi);
    l1Tau.setRawEt(clNxM.rawEt);

    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }

  // Fill output
  iEvent.put(std::move(L1NNCaloTauCollectionBXV), "L1NNCaloTauCollectionBXV");

}  // End of produce function

int l1tNNCaloTauProducer::tower_dIPhi(int& iPhi_1, int& iPhi_2) const {
  const int PI = 36;
  int result = iPhi_1 - iPhi_2;
  if (result > PI) {
    result -= 2 * PI;
  }
  if (result <= -PI) {
    result += 2 * PI;
  }
  return result;
}

int l1tNNCaloTauProducer::tower_dIEta(int& iEta_1, int& iEta_2) const {
  if (iEta_1 * iEta_2 > 0) {
    return iEta_1 - iEta_2;
  } else {
    if (iEta_1 > 0) {
      return iEta_1 - iEta_2 - 1;
    } else {
      return iEta_1 - iEta_2 + 1;
    }
  }
}

int l1tNNCaloTauProducer::endcap_iphi(float& phi) const {
  const float phi_step = 0.0872664;
  if (phi > 0) {
    return floor(phi / phi_step) + 1;
  } else {
    return floor(phi / phi_step) + 73;
  }
}

int l1tNNCaloTauProducer::endcap_ieta(float& eta) const {
  const float eta_step = 0.0845;
  return floor(abs(eta) / eta_step) * std::copysign(1, eta);
}

float l1tNNCaloTauProducer::inputQuantizer(float inputF, float LSB, int nbits) {
  return min(floor(inputF / LSB), float(pow(2, nbits) - 1)) * LSB;
}

float l1tNNCaloTauProducer::inputScaler(float inputF, std::string feature) {
  float mean = FeatScaler_CE.get_child(feature).get<float>("mean");
  float std = FeatScaler_CE.get_child(feature).get<float>("std");

  return (inputF - mean) / std;
}

void l1tNNCaloTauProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // VARIABLES FOR HGCAL PU BDT
  std::vector<edm::ParameterSet> variables;
  edm::ParameterSet set1;
  set1.addParameter<std::string>("name", "eMax");
  set1.addParameter<std::string>("value", "eMax()");
  variables.push_back(set1);
  edm::ParameterSet set2;
  set2.addParameter<std::string>("name", "eMaxOverE");
  set2.addParameter<std::string>("value", "eMax()/energy()");
  variables.push_back(set2);
  edm::ParameterSet set3;
  set3.addParameter<std::string>("name", "sigmaPhiPhiTot");
  set3.addParameter<std::string>("value", "sigmaPhiPhiTot()");
  variables.push_back(set3);
  edm::ParameterSet set4;
  set4.addParameter<std::string>("name", "sigmaRRTot");
  set4.addParameter<std::string>("value", "sigmaRRTot()");
  variables.push_back(set4);
  edm::ParameterSet set5;
  set5.addParameter<std::string>("name", "triggerCells90percent");
  set5.addParameter<std::string>("value", "triggerCells90percent()");
  variables.push_back(set5);

  // // PSET FOR HGCAL PU BDT
  edm::ParameterSetDescription tmp;
  edm::ParameterSetDescription VsPuId;
  VsPuId.addVPSet("variables", tmp, variables);
  VsPuId.add<bool>("isPUFilter", true);
  VsPuId.add<std::string>("preselection", "");
  VsPuId.add<std::string>("method", "BDT");
  VsPuId.add<std::string>("weightsFile", "L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_Pion_vs_Neutrino_BDTweights_1116.xml.gz");
  VsPuId.add<std::string>("wp", "-0.10");

  // DESCRIPTIONS
  edm::ParameterSetDescription desc;
  desc.setComment("Phase2 NN CaloTau (TauMinator) producer plugin.");
  
  desc.add<edm::InputTag>("l1CaloTowers", edm::InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection",""));
  desc.add<edm::InputTag>("hgcalTowers", edm::InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"));
  desc.add<edm::InputTag>("HgcalClusters", edm::InputTag("l1tHGCalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"));

  desc.add<std::string>("preEmId", "hOverE < 0.3 && hOverE >= 0");
  desc.add<edm::ParameterSetDescription>("VsPuId", VsPuId);

  desc.add<double>("EcalEtMinForClustering", 0.);
  desc.add<double>("HcalEtMinForClustering", 0.);
  desc.add<double>("EtMinForSeeding", 2.5);
  desc.add<double>("EtaRestriction", 2.4);
  desc.add<double>("CB_CE_split", 1.55);

  desc.add<std::string>("CNNmodel_CB_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/CNNmodel_CB.pb");
  desc.add<std::string>("DNNident_CB_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNident_CB.pb");
  desc.add<std::string>("DNNcalib_CB_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNcalib_CB.pb");
  desc.add<std::string>("CNNmodel_CE_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/CNNmodel_CE.pb");
  desc.add<std::string>("DNNident_CE_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNident_CE.pb");
  desc.add<std::string>("DNNcalib_CE_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNcalib_CE.pb");
  desc.add<std::string>("FeatScaler_CE_path", "L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/Cl3dFeatScaler_CE.json");

  desc.add<double>("IdWp90_CB", 0.7060);
  desc.add<double>("IdWp95_CB", 0.3432);
  desc.add<double>("IdWp99_CB", 0.0337);
  desc.add<double>("IdWp90_CE", 0.5711);
  desc.add<double>("IdWp95_CE", 0.2742);
  desc.add<double>("IdWp99_CE", 0.0394);

  desc.add<bool>("DEBUG", false);

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(l1tNNCaloTauProducer);