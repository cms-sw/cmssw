// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TInterpreter.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

//L1 trigger Menus etc
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// Jets
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

class IsoTrackCalib : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit IsoTrackCalib(const edm::ParameterSet&);
  ~IsoTrackCalib() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double deltaR(double eta1, double eta2, double phi1, double phi2);

  edm::Service<TFileService> fs_;
  HLTConfigProvider hltConfig_;
  L1GtUtils m_l1GtUtils;
  const L1GtTriggerMenu* m_l1GtMenu;
  const int verbosity_;
  const std::vector<std::string> l1Names_;
  spr::trackSelectionParameters selectionParameters_;
  const std::string theTrackQuality_;
  const double a_coneR_, a_charIsoR_, a_mipR_;
  std::vector<bool>* t_l1bits;

  const edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  const edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  const edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  const edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;
  const edm::EDGetTokenT<reco::GenJetCollection> tok_jets_;
  const edm::EDGetTokenT<reco::PFJetCollection> tok_pfjets_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extTauJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extCenJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extFwdJet_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;

  TTree* tree;
  int t_Run, t_Event, t_ieta;
  double t_EventWeight, t_l1pt, t_l1eta, t_l1phi;
  double t_l3pt, t_l3eta, t_l3phi, t_p, t_mindR1;
  double t_mindR2, t_eMipDR, t_eHcal, t_hmaxNearP;

  bool t_selectTk, t_qltyFlag, t_qltyMissFlag, t_qltyPVFlag;
  std::vector<unsigned int>* t_DetIds;
  std::vector<double>*t_HitEnergies, pbin;
  TProfile *h_RecHit_iEta, *h_RecHit_num;
  TH1I *h_iEta, *h_tkEta0[5], *h_tkEta1[5], *h_tkEta2[5];
  TH1I *h_tkEta3[5], *h_tkEta4[5], *h_tkEta5[5];
  TH1F *h_Rechit_E, *h_jetp;
  TH1F* h_jetpt[4];
  TH1I *h_tketa0[6], *h_tketa1[6], *h_tketa2[6];
  TH1I *h_tketa3[6], *h_tketa4[6], *h_tketa5[6];
  std::map<std::pair<unsigned int, std::string>, int> l1AlgoMap_;
};

static const bool useL1GtTriggerMenuLite(true);
IsoTrackCalib::IsoTrackCalib(const edm::ParameterSet& iConfig)
    : m_l1GtUtils(iConfig, consumesCollector(), useL1GtTriggerMenuLite, *this, L1GtUtils::UseEventSetupIn::Event),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      l1Names_(iConfig.getUntrackedParameter<std::vector<std::string> >("L1Seed")),
      theTrackQuality_(iConfig.getUntrackedParameter<std::string>("TrackQuality", "highPurity")),
      a_coneR_(iConfig.getUntrackedParameter<double>("ConeRadius", 34.98)),
      a_charIsoR_(a_coneR_ + 28.9),
      a_mipR_(iConfig.getUntrackedParameter<double>("ConeRadiusMIP", 14.0)),
      tok_genTrack_(consumes<reco::TrackCollection>(edm::InputTag("generalTracks"))),
      tok_recVtx_(consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"))),
      tok_bs_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))),
      tok_ew_(consumes<GenEventInfoProduct>(edm::InputTag("generatorSmeared"))),
      tok_jets_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("JetSource"))),
      tok_pfjets_(consumes<reco::PFJetCollection>(edm::InputTag("ak5PFJets"))) {
  usesResource(TFileService::kSharedResource);

  //now do whatever initialization is needed
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameters_.minPt = iConfig.getUntrackedParameter<double>("MinTrackPt", 10.0);
  selectionParameters_.minQuality = trackQuality_;
  selectionParameters_.maxDxyPV = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.2);
  selectionParameters_.maxDzPV = iConfig.getUntrackedParameter<double>("MaxDzPV", 5.0);
  selectionParameters_.maxChi2 = iConfig.getUntrackedParameter<double>("MaxChi2", 5.0);
  selectionParameters_.maxDpOverP = iConfig.getUntrackedParameter<double>("MaxDpOverP", 0.1);
  selectionParameters_.minOuterHit = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters_.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters_.maxInMiss = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters_.maxOutMiss = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  bool isItAOD = iConfig.getUntrackedParameter<bool>("IsItAOD", false);
  edm::InputTag L1extraTauJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraTauJetSource");
  edm::InputTag L1extraCenJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraCenJetSource");
  edm::InputTag L1extraFwdJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraFwdJetSource");

  // define tokens for access
  if (isItAOD) {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }
  tok_L1extTauJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraTauJetSource_);
  tok_L1extCenJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraCenJetSource_);
  tok_L1extFwdJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraFwdJetSource_);
  if (verbosity_ >= 0) {
    edm::LogVerbatim("IsoTrack") << "Parameters read from config file \n"
                                 << "\t minPt " << selectionParameters_.minPt << "\t theTrackQuality "
                                 << theTrackQuality_ << "\t minQuality " << selectionParameters_.minQuality
                                 << "\t maxDxyPV " << selectionParameters_.maxDxyPV << "\t maxDzPV "
                                 << selectionParameters_.maxDzPV << "\t maxChi2 " << selectionParameters_.maxChi2
                                 << "\t maxDpOverP " << selectionParameters_.maxDpOverP << "\t minOuterHit "
                                 << selectionParameters_.minOuterHit << "\t minLayerCrossed "
                                 << selectionParameters_.minLayerCrossed << "\t maxInMiss "
                                 << selectionParameters_.maxInMiss << "\t maxOutMiss "
                                 << selectionParameters_.maxOutMiss << "\t a_coneR " << a_coneR_ << "\t a_charIsoR "
                                 << a_charIsoR_ << "\t a_mipR " << a_mipR_ << "\t isItAOD " << isItAOD;
    edm::LogVerbatim("IsoTrack") << l1Names_.size() << " triggers to be studied";
    for (unsigned int k = 0; k < l1Names_.size(); ++k)
      edm::LogVerbatim("IsoTrack") << "[" << k << "]: " << l1Names_[k];
  }

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
}

IsoTrackCalib::~IsoTrackCalib() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrackCalib::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<std::string> seeds = {
      "L1_SingleJet36", "L1_SingleJet52", "L1_SingleJet68", "L1_SingleJet92", "L1_SingleJet128"};
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<std::vector<std::string> >("L1Seed", seeds);
  desc.addUntracked<std::string>("TrackQuality", "highPurity");
  desc.addUntracked<double>("MinTrackPt", 10.0);
  desc.addUntracked<double>("MaxDxyPV", 0.02);
  desc.addUntracked<double>("MaxDzPV", 0.02);
  desc.addUntracked<double>("MaxChi2", 5.0);
  desc.addUntracked<double>("MaxDpOverP", 0.1);
  desc.addUntracked<int>("MinOuterHit", 4);
  desc.addUntracked<int>("MinLayerCrossed", 8);
  desc.addUntracked<int>("MaxInMiss", 0);
  desc.addUntracked<int>("MaxOutMiss", 0);
  desc.addUntracked<double>("ConeRadius", 34.98);
  desc.addUntracked<double>("ConeRadiusMIP", 14.0);
  desc.addUntracked<bool>("IsItAOD", false);
  descriptions.add("isoTrackCalib", desc);
}

void IsoTrackCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  t_Run = iEvent.id().run();
  t_Event = iEvent.id().event();
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Run " << t_Run << " Event " << t_Event << " Luminosity "
                                 << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing()
                                 << " starts ==========";

  //Get magnetic field and geometry
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const MagneticField* bField = &iSetup.getData(tok_magField_);

  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  //event weight for FLAT sample
  t_EventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid())
    t_EventWeight = genEventInfo->weight();

  // genJet information
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(tok_jets_, genJets);
  if (genJets.isValid()) {
    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      const reco::GenJet& genJet = (*genJets)[iGenJet];
      double genJetPt = genJet.pt();
      double genJetEta = genJet.eta();
      h_jetpt[0]->Fill(genJetPt);
      h_jetpt[1]->Fill(genJetPt, t_EventWeight);
      if (genJetEta > -2.5 && genJetEta < 2.5) {
        h_jetpt[2]->Fill(genJetPt);
        h_jetpt[3]->Fill(genJetPt, t_EventWeight);
      }
      break;
    }
  }

  //pf jets
  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(tok_pfjets_, pfJets);

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0, 0, 0);
  if (!recVtxs->empty() && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint((*recVtxs)[0].x(), (*recVtxs)[0].y(), (*recVtxs)[0].z());
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  if (verbosity_ > 10) {
    if ((verbosity_ % 100) / 10 > 2)
      edm::LogVerbatim("IsoTrack") << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid())
      edm::LogVerbatim("IsoTrack") << " Beam Spot " << beamSpotH->position();
  }

  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  HBHERecHitCollection::const_iterator rhitItr;

  for (rhitItr = hbhe->begin(); rhitItr != hbhe->end(); rhitItr++) {
    double rec_energy = rhitItr->energy();
    int rec_ieta = rhitItr->id().ieta();
    int rec_depth = rhitItr->id().depth();
    int rec_zside = rhitItr->id().zside();
    double num1_1 = rec_zside * (rec_ieta + 0.2 * (rec_depth - 1));
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "detid/rechit/ieta/zside/depth/num "
                                   << " = " << rhitItr->id() << "/" << rec_energy << "/" << rec_ieta << "/" << rec_zside
                                   << "/" << rec_depth << "/" << num1_1;
    h_iEta->Fill(rec_ieta);
    h_Rechit_E->Fill(rec_energy);
    h_RecHit_iEta->Fill(rec_ieta, rec_energy);
    h_RecHit_num->Fill(num1_1, rec_energy);
  }

  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, ((verbosity_ / 100) % 10 > 2));
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++) {
    if (trkDetItr->okHCAL) {
      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
      int tk_ieta = detId.ieta();
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      double tk_p = pTrack->p();
      h_tketa0[0]->Fill(tk_ieta);
      for (unsigned int k = 1; k < pbin.size(); ++k) {
        if (tk_p >= pbin[k - 1] && tk_p < pbin[k]) {
          h_tketa0[k]->Fill(tk_ieta);
          break;
        }
      }
    }
  }
  //////////////////////////////L1 Trigger Results//////////////////////////////////////////////////
  t_l1bits->clear();
  for (unsigned int i = 0; i < l1Names_.size(); ++i)
    t_l1bits->push_back(false);
  bool useL1EventSetup = true;
  bool useL1GtTriggerMenuLite = true;

  m_l1GtUtils.getL1GtRunCache(iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);
  int iErrorCode = -1;
  int l1ConfCode = -1;
  const bool l1Conf = m_l1GtUtils.availableL1Configuration(iErrorCode, l1ConfCode);
  if (!l1Conf) {
    edm::LogVerbatim("IsoTrack") << "\nL1 configuration code:" << l1ConfCode
                                 << "\nNo valid L1 trigger configuration available."
                                 << "\nSee text above for error code interpretation"
                                 << "\nNo return here, in order to test each method"
                                 << ", protected against configuration error.";
  }

  const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();
  const std::string& menuName = m_l1GtMenu->gtTriggerMenuName();
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "menuName " << menuName << std::endl;

  std::vector<int> algbits;
  for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
    std::string algName = itAlgo->first;
    int algBitNumber = (itAlgo->second).algoBitNumber();
    bool decision = m_l1GtUtils.decision(iEvent, itAlgo->first, iErrorCode);

    bool l1ok(false);
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << algName << "  " << algBitNumber << "  " << decision;
    for (unsigned int i = 0; i < l1Names_.size(); ++i) {
      if (algName.find(l1Names_[i]) != std::string::npos) {
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "match found"
                                       << " " << algName << "  " << decision;
        t_l1bits->at(i) = (decision > 0);
        if (decision > 0)
          l1ok = true;
      }
    }
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "l1 ok =" << l1ok;

    if (l1ok) {
      edm::Handle<l1extra::L1JetParticleCollection> l1TauHandle;
      iEvent.getByToken(tok_L1extTauJet_, l1TauHandle);
      l1extra::L1JetParticleCollection::const_iterator itr;
      double ptTriggered = -10;
      double etaTriggered = -100;
      double phiTriggered = -100;

      for (itr = l1TauHandle->begin(); itr != l1TauHandle->end(); ++itr) {
        if (itr->pt() > ptTriggered) {
          ptTriggered = itr->pt();
          etaTriggered = itr->eta();
          phiTriggered = itr->phi();
        }
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "tauJ pt " << itr->pt() << "  eta/phi " << itr->eta() << " " << itr->phi();
      }
      edm::Handle<l1extra::L1JetParticleCollection> l1CenJetHandle;
      iEvent.getByToken(tok_L1extCenJet_, l1CenJetHandle);
      for (itr = l1CenJetHandle->begin(); itr != l1CenJetHandle->end(); ++itr) {
        if (itr->pt() > ptTriggered) {
          ptTriggered = itr->pt();
          etaTriggered = itr->eta();
          phiTriggered = itr->phi();
        }
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "cenJ pt     " << itr->pt() << "  eta/phi " << itr->eta() << " "
                                       << itr->phi();
      }
      edm::Handle<l1extra::L1JetParticleCollection> l1FwdJetHandle;
      iEvent.getByToken(tok_L1extFwdJet_, l1FwdJetHandle);
      for (itr = l1FwdJetHandle->begin(); itr != l1FwdJetHandle->end(); ++itr) {
        if (itr->pt() > ptTriggered) {
          ptTriggered = itr->pt();
          etaTriggered = itr->eta();
          phiTriggered = itr->phi();
        }
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "forJ pt     " << itr->pt() << " eta/phi " << itr->eta() << " " << itr->phi();
      }
      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "jets pt/eta/phi = " << ptTriggered << "/" << etaTriggered << "/"
                                     << phiTriggered;
      //////////////////////loop over tracks////////////////////////////////////////
      unsigned int nTracks(0);
      for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
           trkDetItr++, nTracks++) {
        const reco::Track* pTrack = &(*(trkDetItr->trkItr));
        math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());

        t_mindR1 = deltaR(etaTriggered, v4.eta(), phiTriggered, v4.phi());
        t_mindR2 = -999;
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "This track : " << nTracks << " (pt/eta/phi/p) :" << pTrack->pt() << "/"
                                       << pTrack->eta() << "/" << pTrack->phi() << "/" << pTrack->p();

        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "dr values are = " << t_mindR1;

        t_l1pt = ptTriggered;
        t_l1eta = etaTriggered;
        t_l1phi = phiTriggered;
        t_l3pt = -999;
        t_l3eta = -999;
        t_l3phi = -999;

        //Selection of good track
        t_selectTk = spr::goodTrack(pTrack, leadPV, selectionParameters_, ((verbosity_ / 100) % 10 > 2));
        spr::trackSelectionParameters oneCutParameters = selectionParameters_;
        oneCutParameters.maxDxyPV = 10;
        oneCutParameters.maxDzPV = 100;
        oneCutParameters.maxInMiss = 2;
        oneCutParameters.maxOutMiss = 2;
        bool qltyFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));
        oneCutParameters = selectionParameters_;
        oneCutParameters.maxDxyPV = 10;
        oneCutParameters.maxDzPV = 100;
        t_qltyMissFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));
        oneCutParameters = selectionParameters_;
        oneCutParameters.maxInMiss = 2;
        oneCutParameters.maxOutMiss = 2;
        t_qltyPVFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, ((verbosity_ / 100) % 10 > 2));
        t_ieta = 0;
        if (trkDetItr->okHCAL) {
          HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
          t_ieta = detId.ieta();
        }
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" << trkDetItr->okECAL << "/"
                                       << trkDetItr->okHCAL;
        t_qltyFlag = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
        t_p = pTrack->p();
        h_tketa1[0]->Fill(t_ieta);
        for (unsigned int k = 1; k < pbin.size(); ++k) {
          if (t_p >= pbin[k - 1] && t_p < pbin[k]) {
            h_tketa1[k]->Fill(t_ieta);
            break;
          }
        }
        if (t_qltyFlag) {
          h_tketa2[0]->Fill(t_ieta);
          for (unsigned int k = 1; k < pbin.size(); ++k) {
            if (t_p >= pbin[k - 1] && t_p < pbin[k]) {
              h_tketa2[k]->Fill(t_ieta);
              break;
            }
          }
          int nRH_eMipDR(0), nNearTRKs(0), nRecHits(-999);
          t_eMipDR = spr::eCone_ecal(geo,
                                     barrelRecHitsHandle,
                                     endcapRecHitsHandle,
                                     trkDetItr->pointHCAL,
                                     trkDetItr->pointECAL,
                                     a_mipR_,
                                     trkDetItr->directionECAL,
                                     nRH_eMipDR);
          t_DetIds->clear();
          t_HitEnergies->clear();
          std::vector<DetId> ids;
          t_eHcal = spr::eCone_hcal(geo,
                                    hbhe,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_coneR_,
                                    trkDetItr->directionHCAL,
                                    nRecHits,
                                    ids,
                                    *t_HitEnergies);
          for (unsigned int k = 0; k < ids.size(); ++k) {
            t_DetIds->push_back(ids[k].rawId());
          }
          t_hmaxNearP = spr::chargeIsolationCone(
              nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 2));
          if (t_hmaxNearP < 2) {
            h_tketa3[0]->Fill(t_ieta);
            for (unsigned int k = 1; k < pbin.size(); ++k) {
              if (t_p >= pbin[k - 1] && t_p < pbin[k]) {
                h_tketa3[k]->Fill(t_ieta);
                break;
              }
            }
            if (t_eMipDR < 1) {
              h_tketa4[0]->Fill(t_ieta);
              for (unsigned int k = 1; k < pbin.size(); ++k) {
                if (t_p >= pbin[k - 1] && t_p < pbin[k]) {
                  h_tketa4[k]->Fill(t_ieta);
                  break;
                }
              }
              if (t_mindR1 > 1) {
                h_tketa5[0]->Fill(t_ieta);
                for (unsigned int k = 1; k < pbin.size(); ++k) {
                  if (t_p >= pbin[k - 1] && t_p < pbin[k]) {
                    h_tketa5[k]->Fill(t_ieta);
                    break;
                  }
                }
              }
            }
          }
          if (verbosity_ % 10 > 0) {
            edm::LogVerbatim("IsoTrack") << "This track : " << nTracks << " (pt/eta/phi/p) :" << pTrack->pt() << "/"
                                         << pTrack->eta() << "/" << pTrack->phi() << "/" << t_p;
            edm::LogVerbatim("IsoTrack") << "e_MIP " << t_eMipDR << " Chg Isolation " << t_hmaxNearP << " eHcal"
                                         << t_eHcal << " ieta " << t_ieta << " Quality " << t_qltyMissFlag << ":"
                                         << t_qltyPVFlag << ":" << t_selectTk;
            for (unsigned int lll = 0; lll < t_DetIds->size(); lll++) {
              edm::LogVerbatim("IsoTrack") << "det id is = " << t_DetIds->at(lll) << "  "
                                           << " hit enery is  = " << t_HitEnergies->at(lll);
            }
          }
          if (t_p > 20.0 && t_eMipDR < 2.0 && t_hmaxNearP < 10.0) {
            tree->Fill();
          }
        }
      }
    }
  }
}

void IsoTrackCalib::beginJob() {
  h_RecHit_iEta = fs_->make<TProfile>("rechit_ieta", "Rec hit vs. ieta", 60, -30, 30, 0, 1000);
  h_RecHit_num = fs_->make<TProfile>("rechit_num", "Rec hit vs. num", 100, 0, 20, 0, 1000);
  h_iEta = fs_->make<TH1I>("iEta", "iEta", 60, -30, 30);
  h_Rechit_E = fs_->make<TH1F>("Rechit_E", "Rechit_E", 100, 0, 1000);

  double prange[5] = {20, 30, 40, 60, 100};
  for (int k = 0; k < 5; ++k)
    pbin.push_back(prange[k]);
  std::string type[6] = {"All", "Trigger OK", "Tree Selected", "Charge Isolation", "MIP Cut", "L1 Cut"};
  for (unsigned int k = 0; k < pbin.size(); ++k) {
    char name[20], namp[20], title[100];
    if (k == 0)
      sprintf(namp, "all momentum");
    else
      sprintf(namp, "p = %4.0f:%4.0f GeV", pbin[k - 1], pbin[k]);
    sprintf(name, "TrackEta0%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[0].c_str());
    h_tketa0[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
    sprintf(name, "TrackEta1%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[1].c_str());
    h_tketa1[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
    sprintf(name, "TrackEta2%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[2].c_str());
    h_tketa2[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
    sprintf(name, "TrackEta3%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[3].c_str());
    h_tketa3[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
    sprintf(name, "TrackEta4%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[4].c_str());
    h_tketa4[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
    sprintf(name, "TrackEta5%d", k);
    sprintf(title, "Track #eta for tracks with %s (%s)", namp, type[5].c_str());
    h_tketa5[k] = fs_->make<TH1I>(name, title, 60, -30, 30);
  }
  h_jetpt[0] = fs_->make<TH1F>("Jetpt0", "Jet p_T (All)", 500, 0., 2500.);
  h_jetpt[1] = fs_->make<TH1F>("Jetpt1", "Jet p_T (All Weighted)", 500, 0., 2500.);
  h_jetpt[2] = fs_->make<TH1F>("Jetpt2", "Jet p_T (|#eta| < 2.5)", 500, 0., 2500.);
  h_jetpt[3] = fs_->make<TH1F>("Jetpt3", "Jet p_T (|#eta| < 2.5 Weighted)", 500, 0., 2500.);

  tree = fs_->make<TTree>("CalibTree", "CalibTree");

  tree->Branch("t_Run", &t_Run, "t_Run/I");
  tree->Branch("t_Event", &t_Event, "t_Event/I");
  tree->Branch("t_ieta", &t_ieta, "t_ieta/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_l1pt", &t_l1pt, "t_l1pt/D");
  tree->Branch("t_l1eta", &t_l1eta, "t_l1eta/D");
  tree->Branch("t_l1phi", &t_l1phi, "t_l1phi/D");
  tree->Branch("t_l3pt", &t_l3pt, "t_l3pt/D");
  tree->Branch("t_l3eta", &t_l3eta, "t_l3eta/D");
  tree->Branch("t_l3phi", &t_l3phi, "t_l3phi/D");
  tree->Branch("t_p", &t_p, "t_p/D");
  tree->Branch("t_mindR1", &t_mindR1, "t_mindR1/D");
  tree->Branch("t_mindR2", &t_mindR2, "t_mindR2/D");
  tree->Branch("t_eMipDR", &t_eMipDR, "t_eMipDR/D");
  tree->Branch("t_eHcal", &t_eHcal, "t_eHcal/D");
  tree->Branch("t_hmaxNearP", &t_hmaxNearP, "t_hmaxNearP/D");
  tree->Branch("t_selectTk", &t_selectTk, "t_selectTk/O");
  tree->Branch("t_qltyFlag", &t_qltyFlag, "t_qltyFlag/O");
  tree->Branch("t_qltyMissFlag", &t_qltyMissFlag, "t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag", &t_qltyPVFlag, "t_qltyPVFlag/O)");

  t_DetIds = new std::vector<unsigned int>();
  t_HitEnergies = new std::vector<double>();
  t_l1bits = new std::vector<bool>();
  tree->Branch("t_DetIds", "std::vector<unsigned int>", &t_DetIds);
  tree->Branch("t_HitEnergies", "std::vector<double>", &t_HitEnergies);
  tree->Branch("t_l1bits", "std::vector<bool>", &t_l1bits);
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalib::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed = false;
  bool ok = hltConfig_.init(iRun, iSetup, "HLT", changed);
  edm::LogVerbatim("IsoTrack") << "Run " << iRun.run() << " hltconfig.init " << ok;

  int iErrorCode = -1;
  m_l1GtMenu = m_l1GtUtils.ptrL1TriggerMenuEventSetup(iErrorCode);
  const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();
  const std::string& menuName = m_l1GtMenu->gtTriggerMenuName();

  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "menuName " << menuName;
  for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
    std::string algName = itAlgo->first;
    int algBitNumber = (itAlgo->second).algoBitNumber();
    l1AlgoMap_.insert(std::pair<std::pair<unsigned int, std::string>, int>(
        std::pair<unsigned int, std::string>(algBitNumber, algName), 0));
  }
  std::map<std::pair<unsigned int, std::string>, int>::iterator itr;
  for (itr = l1AlgoMap_.begin(); itr != l1AlgoMap_.end(); itr++) {
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << " ********** " << (itr->first).first << " " << (itr->first).second << " "
                                   << itr->second;
  }
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalib::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogVerbatim("IsoTrack") << "endRun " << iRun.run() << std::endl;
}

double IsoTrackCalib::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta() - vec2.eta());
}

double IsoTrackCalib::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  double phi1 = vec1.phi();
  if (phi1 < 0)
    phi1 += 2.0 * M_PI;
  double phi2 = vec2.phi();
  if (phi2 < 0)
    phi2 += 2.0 * M_PI;
  double dphi = phi1 - phi2;
  if (dphi > M_PI)
    dphi -= 2. * M_PI;
  else if (dphi < -M_PI)
    dphi += 2. * M_PI;
  return dphi;
}

double IsoTrackCalib::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  double deta = dEta(vec1, vec2);
  double dphi = dPhi(vec1, vec2);
  return std::sqrt(deta * deta + dphi * dphi);
}

double IsoTrackCalib::deltaR(double eta1, double eta2, double phi1, double phi2) {
  double deta = eta1 - eta2;
  if (phi1 < 0)
    phi1 += 2.0 * M_PI;
  if (phi2 < 0)
    phi2 += 2.0 * M_PI;
  double dphi = phi1 - phi2;
  if (dphi > M_PI)
    dphi -= 2. * M_PI;
  else if (dphi < -M_PI)
    dphi += 2. * M_PI;
  return std::sqrt(deta * deta + dphi * dphi);
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalib);
