/* -*- C++ -*-

Package: L1CaloTrigger
Class: l1tNNCaloTauEmulator
Frinedly name: The TauMinator

\class l1tNNCaloTauEmulator l1tNNCaloTauEmulator.cc

Description: 
Perform firmware-exact emulation of the l1tNNCaloTauProducer
that implements the NN Calo Tau.
(Perform reconstruction and identification of tau 
candidates at L1 Trigger with a CNN.)

Implementation:
The implementation is done forseeing the integration
of the algorithm in the GCT Sum card. This means that
the full detector information can be accessed at the same
time (ECAL, HCAL, HGCAL full eta-phi coverage). This will
come in the form of arrays of towers and clusters.
Given that the emulators of the upstream algortihms are
not fully determined yet, this emulator takes as input
the simulation-based information, manipulates with software
precision to pruduce the arrays of towers and clusters as 
they should be available in the GCT sum card.
Only then the actual fixed point algorithm emulation arrives.

** INFO : THE NNs ARE APPLIED USING THE TENSORFLOW SOFTWARE
          the implementation of full emulation via hls4ml is ongoing
          (it has already been shown in other contexts that tensorflow 
          softwrae and full emulation are very close to each other)

Original Author: Jona Motta
Created: Tue June 7th 2023

*/

#include <iostream>
#include <vector>
#include <cmath>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "ap_int.h"
#include "ap_fixed.h"
// #include "hls4ml/emulator.h"

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

class l1tNNCaloTauEmulator : public edm::stream::EDProducer<> {
public:
  explicit l1tNNCaloTauEmulator(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  // ----fixed LSBs, Nbits, scales, and types----
  static constexpr int INTPHI_PI = 36;
  static constexpr int INTPHI_2PI = 2 * INTPHI_PI;
  static constexpr float IETAPHI_LSB = M_PI / INTPHI_PI;

  static constexpr int FINEINTPHI_PI = 720;
  static constexpr int FINEINTPHI_2PI = 2 * FINEINTPHI_PI;
  static constexpr float ETAPHI_LSB = M_PI / FINEINTPHI_PI;

  static constexpr float SHAPEFEAT_LSB = 0.0000153;  // pow(2, -16)
  static constexpr float SZZ_LSB = SHAPEFEAT_LSB * 100;
  static constexpr float ETAHGCAL_OFFSET = 1.321;  // inferred from hgcal schematics
  static constexpr float IETAHGCAL_LSBp = 0.0808;  // inferred from simulation
  static constexpr float IETAHGCAL_LSB = 0.0845;   // inferred from simulation
  static constexpr float PUID_LSB = 0.00390625;    // pow(2, -8)
  static constexpr float MEANZ_OFFSET = 321.05;    // inferred from hgcal schematics
  static constexpr int IETAHGCAL_OFFSET = 17;
  static constexpr float MEANZ_LSB = 0.5;
  static constexpr float PTET_LSB = 0.25;
  static constexpr float CM2MM = 10;
  static constexpr int R2cone = 0.25 / ETAPHI_LSB / ETAPHI_LSB;

  static constexpr int SHAPEFEAT_W = 16;  // maximum forseen per shape
  static constexpr int DETAPHI_W = 12;
  static constexpr int DIETAPHI_W = 8;
  static constexpr int IETAPHI_W = 7;
  static constexpr int SHOWLEN_W = 6;
  static constexpr int ETAPHI_W = 11;  // precision going to correlator
  static constexpr int MEANZ_W = 12;
  static constexpr int PUID_W = 9;

  static constexpr int PT_W = 14;
  static constexpr int PT_I = 12;
  static constexpr int ET_W = 10;
  static constexpr int ET_I = 8;

  // forseen precision of the HLS4ML emulation of the NNs
  static constexpr int CALIBPT_W = 10;
  static constexpr int CALIBPT_I = 9;
  static constexpr int ID_W = 8;
  static constexpr int ID_I = 1;

  typedef ap_ufixed<PT_W, PT_I, AP_TRN, AP_SAT> Pt_t;
  typedef ap_ufixed<ET_W, ET_I, AP_TRN, AP_SAT> Et_t;

  typedef ap_ufixed<CALIBPT_W, CALIBPT_I, AP_TRN, AP_SAT> CalibPt_t;
  typedef ap_ufixed<ID_W, ID_I, AP_TRN, AP_SAT> Id_t;

  typedef ap_uint<SHAPEFEAT_W> ShapeFeat_t;
  typedef ap_int<DIETAPHI_W> dIEtaPhi_t;
  typedef ap_int<DETAPHI_W> dEtaPhi_t;
  typedef ap_uint<SHOWLEN_W> ShowLen_t;
  typedef ap_int<ETAPHI_W> EtaPhi_t;
  typedef ap_uint<IETAPHI_W> IPhi_t;
  typedef ap_int<IETAPHI_W> IEta_t;
  typedef ap_uint<MEANZ_W> Meanz_t;
  typedef ap_int<PUID_W> PUid_t;

  // ----fixed dimensions of tower clusters----
  const int seedIdx = 22;
  const int IEta_dim = 5;
  const int IPhi_dim = 9;
  const int Eta_limit = 33;

  //----edm control---
  void produce(edm::Event&, const edm::EventSetup&) override;

  //----private functions----
  template <class outPrecision, class inPrecision>
  outPrecision dPhi(inPrecision iPhi_1, inPrecision iPhi_2);
  dIEtaPhi_t tower_dIEta(IEta_t iEta_1, IEta_t iEta_2);
  dEtaPhi_t tw2cl_dPhi(EtaPhi_t iPhi_1, IPhi_t iPhi_2);
  dEtaPhi_t tw2cl_dEta(EtaPhi_t iEta_1, IEta_t iEta_2);
  IEta_t makeEndcapHwIEta(float eta);
  IPhi_t makeEndcapHwIPhi(float phi);
  float apfixedQuantizer(float inputF, float LSB, int nbits);
  int apintQuantizer(float inputF, float LSB, int nbits);
  float inputScaler(float inputF, std::string feature);
  float correctInputEtaCl3d(float eta);
  float correctInputMeanzCl3d(float meanz);

  inline float floatPt(Pt_t pt) { return pt.to_float(); }
  inline float floatEt(Et_t et) { return et.to_float(); }
  inline float floatEta(EtaPhi_t eta) { return eta.to_float() * ETAPHI_LSB; }
  inline float floatPhi(EtaPhi_t phi) { return phi.to_float() * ETAPHI_LSB; }
  inline float floatShape(ShapeFeat_t shape) { return shape.to_float() * SHAPEFEAT_LSB; };
  inline float floatSzz(ShapeFeat_t szz) { return szz.to_float() * SZZ_LSB; };
  inline float floatMeanZ(Meanz_t meanz) { return meanz.to_float() * MEANZ_LSB + MEANZ_OFFSET; };
  inline float floatMeanZHgcalCoord(Meanz_t meanz) { return meanz.to_float() * MEANZ_LSB; };
  inline float floatPuId(PUid_t pu) { return pu.to_float() * PUID_LSB; };
  float floatIEta(IEta_t eta);
  float floatIPhi(IPhi_t phi);

  template <int W>
  ap_int<W> ap_abs(ap_int<W> x);
  template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
  ap_ufixed<W, I> ap_abs(ap_fixed<W, I, _AP_Q, _AP_O> x);

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
  double PuidThr;

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

  PUid_t intPuidThr;
  IEta_t intEtaRestriction;
  IEta_t intCB_CE_split;

  // Class for the towers info as they should be in GCT
  class SimpleTowerHit {
  public:
    IEta_t towerIeta = 0;
    IPhi_t towerIphi = 0;
    Et_t towerEm = 0.;
    Et_t towerHad = 0.;
    Et_t l1egTowerEt = 0.;
    Et_t towerEt = 0.;
    ap_uint<1> isBarrel = 0x1;
    ap_uint<1> stale = 0x0;
    ap_uint<1> stale4seed = 0x0;
  };

  // Class for the clusters info as they should arrive from HGCAL
  class SimpleHGCluster {
  public:
    Pt_t pt;
    EtaPhi_t eta;
    EtaPhi_t phi;
    ShowLen_t showerlength;
    ShowLen_t coreshowerlength;
    ShapeFeat_t spptot;
    ShapeFeat_t szz;
    ShapeFeat_t srrtot;
    Meanz_t meanz;
    PUid_t PUid;
    ap_uint<1> stale = 0x0;
  };

  // Classes for the tower clusters
  class SimplifiedTower {
  public:
    Et_t towerEm = 0.;
    Et_t towerHad = 0.;
    Et_t l1egTowerEt = 0.;

    void fill(SimpleTowerHit Tower) {
      towerEm = Tower.towerEm;
      towerHad = Tower.towerHad;
      l1egTowerEt = Tower.l1egTowerEt;
    }
  };

  class InputTowerCluster {
  public:
    SimplifiedTower towerHits[45];
    ap_uint<1> barrelSeeded = 0x0;
    ap_uint<1> filled[45];

    void fill(int idx, SimpleTowerHit Tower) {
      towerHits[idx].fill(Tower);
      filled[idx] = 0x1;
    }

    void init() {
      SimplifiedTower emptyT;
      std::fill(towerHits, towerHits + 44, emptyT);
      std::fill(filled, filled + 44, 0x0);
    }
  };

  class InputTowerCluster_pstn {
  public:
    IEta_t seedIeta = 0;
    IPhi_t seedIphi = 0;

    void fill(SimpleTowerHit Tower) {
      seedIeta = Tower.towerIeta;
      seedIphi = Tower.towerIphi;
    }
  };

  // INFO : now variables are in GCT precision, they should be in NN precision
  // after scaling, i.e. something like ap_fixed<16, 6, AP_TRN, AP_SAT>, when the
  // full hls4ml emulation is available
  class InputHGCluster {
  public:
    Pt_t pt;
    EtaPhi_t eta;
    ShowLen_t showerlength;
    ShowLen_t coreshowerlength;
    ShapeFeat_t spptot;
    ShapeFeat_t szz;
    ShapeFeat_t srrtot;
    Meanz_t meanz;

    void fill(SimpleHGCluster Cluster) {
      pt = Cluster.pt;
      eta = Cluster.eta;
      showerlength = Cluster.showerlength;
      coreshowerlength = Cluster.coreshowerlength;
      spptot = Cluster.spptot;
      szz = Cluster.szz;
      srrtot = Cluster.srrtot;
      meanz = Cluster.meanz;
    }
  };

  l1t::Tau MakeTauCandidate(bool isBarrel, int clNxMIdx, std::vector<tensorflow::Tensor> outputsIdent, std::vector<tensorflow::Tensor> outputsCalib, std::vector<InputTowerCluster_pstn> clustersNxM_pstn);
};

/*
████████ ██   ██ ██████     ████████  █████  ██   ██ ███    ███ ██ ███    ██  █████  ████████  ██████  ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ████  ████ ██ ████   ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ███████ █████         ██    ███████ ██   ██ ██ ████ ██ ██ ██ ██  ██ ███████    ██    ██    ██ ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ██  ██  ██ ██ ██  ██ ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ██   ██ ██████        ██    ██   ██ ███████ ██      ██ ██ ██   ████ ██   ██    ██     ██████  ██    ██
*/

// ----Constructor and Destructor -----
l1tNNCaloTauEmulator::l1tNNCaloTauEmulator(const edm::ParameterSet& iConfig)
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
      PuidThr(iConfig.getParameter<double>("PuidThr")),

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

  intPuidThr = apintQuantizer(PuidThr, PUID_LSB, PUID_W);
  intEtaRestriction = apintQuantizer(EtaRestriction, IETAPHI_LSB, IETAPHI_W);
  intCB_CE_split = apintQuantizer(CB_CE_split, IETAPHI_LSB, IETAPHI_W) + 1;

  // Create produced outputs
  produces<BXVector<l1t::Tau>>("L1NNCaloTauCollectionBXV");

  // Settings output
  edm::LogInfo("Settings") << "EtaRestriction = " << EtaRestriction << " (" << intEtaRestriction << ")"
                           << " , CB_CE_split = " << CB_CE_split << "(" << intCB_CE_split
                           << ") , EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering
                           << " , EcalTpEtMin = " << EcalEtMinForClustering << " , PuidThr = " << PuidThr << "(" << intPuidThr << ")"
                           << std::endl;
}

void l1tNNCaloTauEmulator::produce(edm::Event& iEvent, const edm::EventSetup& eSetup) {
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
    l1Hit.isBarrel = 0x1;
    l1Hit.l1egTowerEt = apfixedQuantizer(hit.l1egTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerEm = apfixedQuantizer(hit.ecalTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerHad = apfixedQuantizer(hit.hcalTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerEt = apfixedQuantizer(hit.ecalTowerEt() + hit.hcalTowerEt() + hit.l1egTowerEt(), PTET_LSB, ET_W);
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
    l1Hit.isBarrel = 0x0;
    l1Hit.l1egTowerEt = 0.0;
    l1Hit.towerEm = apfixedQuantizer(hit.etEm(), PTET_LSB, ET_W);
    l1Hit.towerHad = apfixedQuantizer(hit.etHad(), PTET_LSB, ET_W);
    l1Hit.towerEt = apfixedQuantizer(hit.etEm() + hit.etHad(), PTET_LSB, ET_W);
    l1Hit.towerIeta = makeEndcapHwIEta(hit.eta());
    l1Hit.towerIphi = makeEndcapHwIPhi(hit.phi());

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

    float idScore = -1.;
    if (!VsPuId.method().empty()) {
      int id = VsPuId.passID(*cl3dIt, cluster);
      idScore = cluster.egVsPUMVAOut();
    }

    float eta_hgcalCoord = correctInputEtaCl3d(cl3d.eta());
    float meanz_hgcalCoord = correctInputMeanzCl3d(cl3d.zBarycenter());

    SimpleHGCluster HGCluster;
    HGCluster.pt = apfixedQuantizer(cl3d.pt(), PTET_LSB, PT_W);
    HGCluster.eta = apintQuantizer(eta_hgcalCoord, ETAPHI_LSB, ETAPHI_W);
    HGCluster.phi = apintQuantizer(cl3d.phi(), ETAPHI_LSB, ETAPHI_W);
    HGCluster.showerlength = cl3d.showerLength();
    HGCluster.coreshowerlength = cl3d.coreShowerLength();
    HGCluster.spptot = apintQuantizer(cl3d.sigmaPhiPhiTot(), SHAPEFEAT_LSB, SHAPEFEAT_W);
    HGCluster.szz = apintQuantizer(cl3d.sigmaZZ(), SZZ_LSB, SHAPEFEAT_W);
    HGCluster.srrtot = apintQuantizer(cl3d.sigmaRRTot(), SHAPEFEAT_LSB, SHAPEFEAT_W);
    HGCluster.meanz = apintQuantizer(meanz_hgcalCoord, MEANZ_LSB, MEANZ_W);
    HGCluster.PUid = apintQuantizer(idScore, PUID_LSB, PUID_W);

    AllHGClusters.push_back(HGCluster);
  }

  // Order the collection in pt (the input to the GCT will be pt ordered)
  std::sort(begin(AllHGClusters), end(AllHGClusters), [](const SimpleHGCluster& a, SimpleHGCluster& b) {
    return a.pt > b.pt;
  });

  /*
  // END OF SOFTWARE PRECISION SECTION
  // up to here treated inputs from simulation with SW precision
  // to massage them into the HW precision varibales as they are
  // forseen (roughly) to be available at the GCT Sum card level
  // ------------------------------------------------------------- */

  // Make NxM TowerClusters and HGClusters collections for TauMinator
  std::vector<InputTowerCluster> l1TowerClustersNxM_CB;
  std::vector<InputTowerCluster_pstn> l1TowerClustersNxM_CB_pstn;
  std::vector<InputTowerCluster> l1TowerClustersNxM_CE;
  std::vector<InputTowerCluster_pstn> l1TowerClustersNxM_CE_pstn;
  std::vector<InputHGCluster> HGClusters;

  // Supporting collection of endcap clusters before cl3d matching
  std::vector<InputTowerCluster> AllL1TowerClustersNxM_CE;
  std::vector<InputTowerCluster_pstn> AllL1TowerClustersNxM_CE_pstn;

  int Nclusters_CB = 0;
  int AllNclusters_CE = 0;
  bool caloTauSeedingFinished = false;
  // Loop for seeding of clNxM objects
  while (!caloTauSeedingFinished) {
    InputTowerCluster clNxM;
    clNxM.init();
    InputTowerCluster_pstn clNxM_pstn;
    bool seeded = false;

    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip seeding in towers that would make the cluster extend in HF
      // Skip l1CaloTowers which are already used by this clusters' mask
      if (ap_abs(l1CaloTower.towerIeta) > Eta_limit || ap_abs(l1CaloTower.towerIeta) > intEtaRestriction ||
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

        clNxM.fill(seedIdx, l1CaloTower);
        clNxM_pstn.fill(l1CaloTower);
        if (l1CaloTower.isBarrel) {
          clNxM.barrelSeeded = 0x1;
        }

        l1CaloTower.stale4seed = 0x1;
        l1CaloTower.stale = 0x1;
        seeded = true;

        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM_pstn.seedIeta);
      dIEtaPhi_t d_iPhi = dPhi<dIEtaPhi_t,IPhi_t>(l1CaloTower.towerIphi, clNxM_pstn.seedIphi);

      // Stale tower for seeding if it would lead to overalp between clusters
      if ((ap_abs(d_iEta) <= IEta_dim - 1 && ap_abs(d_iPhi) <= IPhi_dim - 1)) {
        l1CaloTower.stale4seed = 0x1;
      }

    }  // End for loop over TPs

    // Pushback seeds split in barrel and endcap
    if (seeded) {
      if (ap_abs(clNxM_pstn.seedIeta) <= intCB_CE_split) {
        l1TowerClustersNxM_CB.push_back(clNxM);
        l1TowerClustersNxM_CB_pstn.push_back(clNxM_pstn);
        Nclusters_CB++;
      } else {
        AllL1TowerClustersNxM_CE.push_back(clNxM);
        AllL1TowerClustersNxM_CE_pstn.push_back(clNxM_pstn);
        AllNclusters_CE++;
      }
    }

  }  // End while loop of TowerClusters seeding

  // Loop for barrel NxM TowerClusters clustering starting from the seeds
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {
    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIeta);
      dIEtaPhi_t d_iPhi = dPhi<dIEtaPhi_t,IPhi_t>(l1CaloTower.towerIphi, l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIphi);
      int hitIdx = d_iEta * 9 + d_iPhi + seedIdx;

      // Cluster all towers in a NxM towers mask
      if ((ap_abs(d_iEta) <= (IEta_dim - 1) / 2 && ap_abs(d_iPhi) <= (IPhi_dim - 1) / 2)) {
        l1TowerClustersNxM_CB[clNxMIdx].fill(hitIdx, l1CaloTower);
        l1CaloTower.stale = 0x1;
      }

    }  // End for loop over TPs

  }  // End while loop of barrel TowerClusters creation

  // In the endcap cross-loop over clNxM and cl3d to match them
  // (we can do it before full clustering just using the seed info)
  int Nclusters_CE = 0;
  for (int clNxMIdx = 0; clNxMIdx < AllNclusters_CE; clNxMIdx++) {
    bool matched = false;
    for (auto& HGCluster : AllHGClusters) {
      // In case the clNxM or HGCluster have already been matched just continue through the list to the end
      // only use cl3ds above 4GeV and above -0.10 pu id
      if (matched || HGCluster.stale || HGCluster.pt < Pt_t(4.) || HGCluster.PUid < intPuidThr) {
        continue;
      }

      dEtaPhi_t d_iEta = tw2cl_dEta(HGCluster.eta, AllL1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
      dEtaPhi_t d_iPhi = tw2cl_dPhi(HGCluster.phi, AllL1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);
      matched = d_iEta * d_iEta + d_iPhi * d_iPhi < R2cone;

      if (matched) {
        HGCluster.stale = 0x1;
        InputHGCluster cl3d;
        cl3d.fill(HGCluster);
        HGClusters.push_back(cl3d);
        l1TowerClustersNxM_CE.push_back(AllL1TowerClustersNxM_CE[clNxMIdx]);
        l1TowerClustersNxM_CE_pstn.push_back(AllL1TowerClustersNxM_CE_pstn[clNxMIdx]);
        Nclusters_CE++;
      }

    }  // End for loop over cl3ds

  }  // End for loop over clNxM

  // Loop for endcap matched NxM TowerClusters clustering starting from the seeds just found
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {
    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
      dIEtaPhi_t d_iPhi = dPhi<dIEtaPhi_t,IPhi_t>(l1CaloTower.towerIphi, l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);
      int hitIdx = d_iEta * 9 + d_iPhi + seedIdx;

      // Cluster all towers in a NxM towers mask
      if ((ap_abs(d_iEta) <= (IEta_dim - 1) / 2 && ap_abs(d_iPhi) <= (IPhi_dim - 1) / 2)) {
        l1TowerClustersNxM_CE[clNxMIdx].fill(hitIdx, l1CaloTower);
        l1CaloTower.stale = 0x1;
      }

    }  // End for loop over TPs

  }  // End while loop of barrel TowerClusters creation

  // Barrel TauMinator application
  tensorflow::setLogging("2");
  int batchSize_CB = (int)(Nclusters_CB);
  tensorflow::TensorShape imageShape_CB({batchSize_CB, IEta_dim, IPhi_dim, 2});
  tensorflow::TensorShape positionShape_CB({batchSize_CB, 2});
  tensorflow::Tensor TowerClusterImage_CB(tensorflow::DT_FLOAT, imageShape_CB);
  tensorflow::Tensor TowerClusterPosition_CB(tensorflow::DT_FLOAT, positionShape_CB);

  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {
    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CB.tensor<float, 4>()(clNxMIdx, eta, phi, 0) =
            (l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].l1egTowerEt.to_float() +
             l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].towerEm.to_float());
        TowerClusterImage_CB.tensor<float, 4>()(clNxMIdx, eta, phi, 1) =
            (l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].towerHad.to_float());
      }
    }

    TowerClusterPosition_CB.tensor<float, 2>()(clNxMIdx, 0) = floatIEta(l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIeta);
    TowerClusterPosition_CB.tensor<float, 2>()(clNxMIdx, 1) = floatIPhi(l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIphi);
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
  tensorflow::run(DNNcalib_CBsession, DNN_CBinputsList, {"TauMinator_CB_calib/DNNout/MatMul"}, &DNN_CBoutputsCalib);

  // Endcap TauMinator application
  int batchSize_CE = (int)(Nclusters_CE);
  tensorflow::TensorShape imageShape_CE({batchSize_CE, IEta_dim, IPhi_dim, 2});
  tensorflow::TensorShape positionShape_CE({batchSize_CE, 2});
  tensorflow::TensorShape cl3dfeatShape_CE({batchSize_CE, 8});
  tensorflow::Tensor TowerClusterImage_CE(tensorflow::DT_FLOAT, imageShape_CE);
  tensorflow::Tensor TowerClusterPosition_CE(tensorflow::DT_FLOAT, positionShape_CE);
  tensorflow::Tensor Cl3dShapeFeatures_CE(tensorflow::DT_FLOAT, cl3dfeatShape_CE);

  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {
    // Indexing of cl3ds is the same as the one of clNxMs
    InputHGCluster HGClu = HGClusters[clNxMIdx];

    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CE.tensor<float, 4>()(clNxMIdx, eta, phi, 0) =
            (l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].l1egTowerEt.to_float() +
             l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].towerEm.to_float());
        TowerClusterImage_CE.tensor<float, 4>()(clNxMIdx, eta, phi, 1) =
            (l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].towerHad.to_float());
      }
    }

    TowerClusterPosition_CE.tensor<float, 2>()(clNxMIdx, 0) = floatIEta(l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
    TowerClusterPosition_CE.tensor<float, 2>()(clNxMIdx, 1) = floatIPhi(l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);

    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 0) = inputScaler(HGClu.pt.to_float(), "pt");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 1) = inputScaler(abs(floatEta(HGClu.eta)), "eta");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 2) = inputScaler(HGClu.showerlength.to_float(), "showerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 3) =
        inputScaler(HGClu.coreshowerlength.to_float(), "coreshowerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 4) = inputScaler(floatShape(HGClu.spptot), "spptot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 5) = inputScaler(floatSzz(HGClu.szz), "szz");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 6) = inputScaler(floatShape(HGClu.srrtot), "srrtot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 7) = inputScaler(floatMeanZHgcalCoord(HGClu.meanz), "meanz");
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

  // ------------------------------------------------------------- */
  // RESTART OF SOFTWARE PRECISION SECTION
  // from here on we go back to floating point precision to
  // produce the output for Ntuplization and further work,
  // and the output for the GT.
  // *

  // Fill the output collection of L1 taus with the barrel candidates
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {
    l1t::Tau l1Tau = MakeTauCandidate(true, clNxMIdx, DNN_CBoutputsIdent, DNN_CBoutputsCalib, l1TowerClustersNxM_CB_pstn);
    if (l1Tau.pt()<0) { continue; }
    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }

  // Fill the output collection of L1 taus with the endcap candidates
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {
    l1t::Tau l1Tau = MakeTauCandidate(false, clNxMIdx, DNN_CEoutputsIdent, DNN_CEoutputsCalib, l1TowerClustersNxM_CE_pstn);
    if (l1Tau.pt()<0) { continue; }
    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }

  // Fill output
  iEvent.put(std::move(L1NNCaloTauCollectionBXV), "L1NNCaloTauCollectionBXV");

}  // End of produce function

template <class outPrecision, class inPrecision>
outPrecision l1tNNCaloTauEmulator::dPhi(inPrecision iPhi_1, inPrecision iPhi_2) {
  outPrecision dphi = iPhi_1 - iPhi_2;

  outPrecision dphi0 = dphi > outPrecision(INTPHI_PI) ? outPrecision(dphi - INTPHI_2PI) : dphi;
  outPrecision dphi1 = dphi <= outPrecision(-INTPHI_PI) ? outPrecision(dphi + INTPHI_2PI) : dphi;

  outPrecision result = dphi > outPrecision(0) ? dphi0 : dphi1;

  return result;
}

l1tNNCaloTauEmulator::dIEtaPhi_t l1tNNCaloTauEmulator::tower_dIEta(IEta_t iEta_1, IEta_t iEta_2) {
  ap_int<12> mult = iEta_1 * iEta_2;
  dIEtaPhi_t result = iEta_1 - iEta_2;
  if (mult < 0) {
    result = iEta_1 > 0 ? result - 1 : result + 1;
  }

  return result;
}

l1tNNCaloTauEmulator::dEtaPhi_t l1tNNCaloTauEmulator::tw2cl_dPhi(EtaPhi_t iPhi_1, IPhi_t iPhi_2) {
  EtaPhi_t shiftediPhi_2 = iPhi_2 <= IPhi_t(36) ? EtaPhi_t(iPhi_2) : EtaPhi_t(iPhi_2 - INTPHI_2PI + 1);

  EtaPhi_t fineiPhi_2 = shiftediPhi_2 * (IETAPHI_LSB / ETAPHI_LSB);
  // subrtaction of half rescaling corrects from edge to center of tower
  fineiPhi_2 = fineiPhi_2 > EtaPhi_t(0) ? EtaPhi_t(fineiPhi_2 - (IETAPHI_LSB / ETAPHI_LSB) / 2)
                                        : EtaPhi_t(fineiPhi_2 + (IETAPHI_LSB / ETAPHI_LSB) / 2);

  return dPhi<dEtaPhi_t,EtaPhi_t>(iPhi_1, fineiPhi_2);
}

l1tNNCaloTauEmulator::dEtaPhi_t l1tNNCaloTauEmulator::tw2cl_dEta(EtaPhi_t iEta_1, IEta_t iEta_2) {
  // change from hgcal frame to barrel-centered frame
  EtaPhi_t framechangeCl3d = 303;  // 303*pi/720 = 1.322
  iEta_1 = iEta_1 > EtaPhi_t(0) ? EtaPhi_t(iEta_1 + framechangeCl3d) : EtaPhi_t(iEta_1 - framechangeCl3d);

  // the actual depth is 330 but 329 corrects for 0.0808 tower
  EtaPhi_t barrelEtaDepth = 329;
  EtaPhi_t fineiEta_2 = barrelEtaDepth + (iEta_2 - IETAHGCAL_OFFSET) * (IETAHGCAL_LSB / ETAPHI_LSB);

  return iEta_1 - fineiEta_2;
}

l1tNNCaloTauEmulator::IEta_t l1tNNCaloTauEmulator::makeEndcapHwIEta(float eta) {
  IEta_t ieta = floor(eta / IETAHGCAL_LSB);
  // +1 because flooring gets it 1 unit lower when negative
  ieta = ieta < IEta_t(0) ? IEta_t(ieta + 1) : ieta;

  return ieta;
}

l1tNNCaloTauEmulator::IPhi_t l1tNNCaloTauEmulator::makeEndcapHwIPhi(float phi) {
  phi = phi < 0 ? phi + 2 * M_PI : phi;

  // +1 because tower 0 does not exist
  return floor(phi / IETAPHI_LSB) + 1;
}

template <int W>
ap_int<W> l1tNNCaloTauEmulator::ap_abs(ap_int<W> x) {
  ap_int<W> result;
  if (x < 0) {
    result = -x;
  } else {
    result = x;
  }

  return result;
}

template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
ap_ufixed<W, I> l1tNNCaloTauEmulator::ap_abs(ap_fixed<W, I, _AP_Q, _AP_O> x) {
  ap_ufixed<W, I> result;
  if (x < 0) {
    result = -x;
  } else {
    result = x;
  }

  return result;
}

float l1tNNCaloTauEmulator::apfixedQuantizer(float inputF, float LSB, int nbits) {
  return min(floor(inputF / LSB), float(pow(2, nbits) - 1)) * LSB;
}

int l1tNNCaloTauEmulator::apintQuantizer(float inputF, float LSB, int nbits) {
  return min(floor(inputF / LSB), float(pow(2, nbits) - 1));
}

float l1tNNCaloTauEmulator::inputScaler(float inputF, std::string feature) {
  float mean = FeatScaler_CE.get_child(feature).get<float>("mean");
  float std = FeatScaler_CE.get_child(feature).get<float>("std");

  return (inputF - mean) / std;
}

float l1tNNCaloTauEmulator::correctInputEtaCl3d(float eta) {
  return eta > 0 ? eta - ETAHGCAL_OFFSET : eta + ETAHGCAL_OFFSET;
}

float l1tNNCaloTauEmulator::correctInputMeanzCl3d(float meanz) { return CM2MM * (abs(meanz) - MEANZ_OFFSET); }

float l1tNNCaloTauEmulator::floatIEta(IEta_t eta) {
  // transform eta of towers from integer to float, correcting for different barrel/endcap LSB
  float feta;
  if (abs(eta) > IETAHGCAL_OFFSET) {
    if (eta > 0) {
      feta = IETAHGCAL_OFFSET * IETAPHI_LSB - (IETAHGCAL_LSB - IETAHGCAL_LSBp) +
             (eta.to_float() - IETAHGCAL_OFFSET) * IETAHGCAL_LSB;
    } else {
      feta = -IETAHGCAL_OFFSET * IETAPHI_LSB + (IETAHGCAL_LSB - IETAHGCAL_LSBp) +
             (eta.to_float() + IETAHGCAL_OFFSET) * IETAHGCAL_LSB;
    }
  } else {
    feta = eta.to_float() * IETAPHI_LSB;
  }

  // shift by half a tower to consider the tower center instead of the edge
  return feta > 0 ? feta - IETAPHI_LSB / 2 : feta + IETAPHI_LSB / 2;
}

float l1tNNCaloTauEmulator::floatIPhi(IPhi_t phi) {
  float fphi = phi.to_float();
  // add 2pi + 1 because tower 0 does not exist
  fphi = fphi > INTPHI_PI ? fphi - INTPHI_2PI + 1 : fphi;
  fphi *= IETAPHI_LSB;

  // shift by half a tower to consider the tower center instead of the edge
  return fphi > 0 ? fphi - IETAPHI_LSB / 2 : fphi + IETAPHI_LSB / 2;
}

l1t::Tau l1tNNCaloTauEmulator::MakeTauCandidate(bool isBarrel, int clNxMIdx, std::vector<tensorflow::Tensor> outputsIdent, std::vector<tensorflow::Tensor> outputsCalib, std::vector<l1tNNCaloTauEmulator::InputTowerCluster_pstn> clustersNxM_pstn) {
    int seedIeta = clustersNxM_pstn[clNxMIdx].seedIeta;
    int seedIphi = clustersNxM_pstn[clNxMIdx].seedIphi;

    if (seedIeta > intEtaRestriction) {
      return l1t::Tau(reco::Candidate::PolarLorentzVector(-1, 0, 0, 0), -1, 0, 0, 0, 0);;
    }

    float tau_IDscore = outputsIdent[0].matrix<float>()(0, clNxMIdx);
    float tau_calibPt = outputsCalib[0].matrix<float>()(0, clNxMIdx);
    float tau_eta = floatIEta(seedIeta);
    float tau_phi = floatIPhi(seedIphi);

    // Assign increasing quality to higher scoring candidates
    int quality = 0;
    if (isBarrel) {
      // 99% WP
      if (tau_IDscore > IdWp99_CB) {
        quality = 1;
      }
      // 95% WP
      if (tau_IDscore > IdWp95_CB) {
        quality = 2;
      }
      // 90% WP
      if (tau_IDscore > IdWp90_CB) {
        quality = 3;
      }
    }
    else {
      // 99% WP
      if (tau_IDscore > IdWp99_CE) {
        quality = 1;
      }
      // 95% WP
      if (tau_IDscore > IdWp95_CE) {
        quality = 2;
      }
      // 90% WP
      if (tau_IDscore > IdWp90_CE) {
        quality = 3;
      }
    }

    reco::Candidate::PolarLorentzVector tauP4 = reco::Candidate::PolarLorentzVector(tau_calibPt, tau_eta, tau_phi, 0);

    // store ID score multiplied by 10E4 to have good precision even using the Phase1 tau int iso format
    // (this is stored just in case for possible additional offline studies)
    // tau initialisation =  (p4,    pt,          eta,     phi,     qual,    iso)
    l1t::Tau l1Tau = l1t::Tau(tauP4, tau_calibPt, tau_eta, tau_phi, quality, tau_IDscore * 10E4);
    l1Tau.setTowerIEta(seedIeta);
    l1Tau.setTowerIPhi(seedIphi);

    return l1Tau;
}

void l1tNNCaloTauEmulator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
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
  desc.add<double>("PuidThr", -0.10);

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

DEFINE_FWK_MODULE(l1tNNCaloTauEmulator);
