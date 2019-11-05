#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include "TPRegexp.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

//////////////trigger info////////////////////////////////////

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#define EDM_ML_DEBUG

class HcalHBHEMuonHighEtaAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalHBHEMuonHighEtaAnalyzer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  bool analyzeMuon(edm::Event const&, math::XYZPoint&);
  bool analyzeHadron(edm::Event const&, math::XYZPoint&);
  bool analyzeTracks(const reco::Track*, math::XYZPoint&, int, std::vector<spr::propagatedTrackID>&, bool);
  void clearVectors();
  int matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&);
  bool isGoodVertex(const reco::Vertex&);
  double respCorr(const DetId&);
  double gainFactor(const edm::ESHandle<HcalDbService>&, const HcalDetId&);
  int depth16HE(int, int);
  bool goodCell(const HcalDetId&, const reco::Track*, const CaloGeometry*, const MagneticField*);
  void fillTrackParameters(const reco::Track*, math::XYZPoint);

  // ----------member data ---------------------------
  const edm::InputTag labelEBRecHit_, labelEERecHit_, labelHBHERecHit_;
  const std::string labelVtx_, labelMuon_, labelGenTrack_;
  const double etaMin_, emaxNearPThr_;
  const bool analyzeMuon_, unCorrect_, collapseDepth_, isItPlan1_, getCharge_;
  const int useRaw_, verbosity_;
  const std::string theTrackQuality_, fileInCorr_;
  const bool ignoreHECorr_, isItPreRecHit_, writeRespCorr_;
  bool mergedDepth_, useMyCorr_;
  int maxDepth_, kount_;
  spr::trackSelectionParameters selectionParameter_;

  const HcalDDDRecConstants* hdc_;
  const HcalTopology* theHBHETopology_;
  HcalRespCorrs* respCorrs_;

  edm::EDGetTokenT<reco::VertexCollection> tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;

  edm::ESHandle<CaloGeometry> pG_;
  edm::ESHandle<MagneticField> bFieldH_;
  edm::ESHandle<EcalChannelStatus> ecalChStatus_;
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv_;
  edm::ESHandle<CaloTopology> theCaloTopology_;
  edm::ESHandle<HcalDbService> conditions_;

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle_;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle_;
  edm::Handle<HBHERecHitCollection> hbhe_;

  //////////////////////////////////////////////////////
  static const int depthMax_ = 7;
  TTree* tree_;
  unsigned int runNumber_, eventNumber_, goodVertex_;
  std::vector<bool> mediumMuon_;
  std::vector<double> ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  std::vector<double> isolationR04_, isolationR03_;
  std::vector<double> ecalEnergy_, hcalEnergy_, hoEnergy_;
  std::vector<bool> matchedId_, hcalHot_;
  std::vector<double> ecal3x3Energy_, hcal1x1Energy_;
  std::vector<unsigned int> ecalDetId_, hcalDetId_, ehcalDetId_;
  std::vector<int> hcal_ieta_, hcal_iphi_;
  std::vector<double> hcalDepthEnergy_[depthMax_];
  std::vector<double> hcalDepthActiveLength_[depthMax_];
  std::vector<double> hcalDepthEnergyHot_[depthMax_];
  std::vector<double> hcalDepthActiveLengthHot_[depthMax_];
  std::vector<double> hcalDepthChargeHot_[depthMax_];
  std::vector<double> hcalDepthChargeHotBG_[depthMax_];
  std::vector<double> hcalDepthEnergyCorr_[depthMax_];
  std::vector<double> hcalDepthEnergyHotCorr_[depthMax_];
  std::vector<bool> hcalDepthMatch_[depthMax_];
  std::vector<bool> hcalDepthMatchHot_[depthMax_];
  std::vector<double> hcalActiveLength_, hcalActiveLengthHot_;
  std::vector<double> emaxNearP_, trackDz_;
  std::vector<int> trackLayerCrossed_, trackOuterHit_;
  std::vector<int> trackMissedInnerHits_, trackMissedOuterHits_;

  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB, actHE;
  std::map<DetId, double> corrValue_;
  ////////////////////////////////////////////////////////////
};

HcalHBHEMuonHighEtaAnalyzer::HcalHBHEMuonHighEtaAnalyzer(const edm::ParameterSet& iConfig)
    : labelEBRecHit_(iConfig.getParameter<edm::InputTag>("labelEBRecHit")),
      labelEERecHit_(iConfig.getParameter<edm::InputTag>("labelEERecHit")),
      labelHBHERecHit_(iConfig.getParameter<edm::InputTag>("labelHBHERecHit")),
      labelVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelMuon_(iConfig.getParameter<std::string>("labelMuon")),
      labelGenTrack_(iConfig.getParameter<std::string>("labelTrack")),
      etaMin_(iConfig.getParameter<double>("etaMin")),
      emaxNearPThr_(iConfig.getParameter<double>("emaxNearPThreshold")),
      analyzeMuon_(iConfig.getParameter<bool>("analyzeMuon")),
      unCorrect_(iConfig.getParameter<bool>("unCorrect")),
      collapseDepth_(iConfig.getParameter<bool>("collapseDepth")),
      isItPlan1_(iConfig.getParameter<bool>("isItPlan1")),
      getCharge_(iConfig.getParameter<bool>("getCharge")),
      useRaw_(iConfig.getParameter<int>("useRaw")),
      verbosity_(iConfig.getParameter<int>("verbosity")),
      theTrackQuality_(iConfig.getUntrackedParameter<std::string>("trackQuality")),
      fileInCorr_(iConfig.getUntrackedParameter<std::string>("fileInCorr", "")),
      ignoreHECorr_(iConfig.getUntrackedParameter<bool>("ignoreHECorr", false)),
      isItPreRecHit_(iConfig.getUntrackedParameter<bool>("isItPreRecHit", false)),
      writeRespCorr_(iConfig.getUntrackedParameter<bool>("writeRespCorr", false)),
      hdc_(nullptr),
      theHBHETopology_(nullptr),
      respCorrs_(nullptr),
      tree_(nullptr) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  kount_ = 0;
  maxDepth_ = iConfig.getUntrackedParameter<int>("maxDepth", 7);
  if (maxDepth_ > depthMax_)
    maxDepth_ = depthMax_;
  else if (maxDepth_ < 1)
    maxDepth_ = 4;

  reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt = iConfig.getUntrackedParameter<double>("minTrackPt");
  selectionParameter_.minQuality = trackQuality;
  selectionParameter_.maxDxyPV = iConfig.getUntrackedParameter<double>("maxDxyPV");
  selectionParameter_.maxDzPV = iConfig.getUntrackedParameter<double>("maxDzPV");
  selectionParameter_.maxChi2 = iConfig.getUntrackedParameter<double>("maxChi2");
  selectionParameter_.maxDpOverP = iConfig.getUntrackedParameter<double>("maxDpOverP");
  selectionParameter_.minOuterHit = selectionParameter_.minLayerCrossed = 0;
  selectionParameter_.maxInMiss = selectionParameter_.maxOutMiss = 2;

  mergedDepth_ = (!isItPreRecHit_) || (collapseDepth_);
  tok_EB_ = consumes<EcalRecHitCollection>(labelEBRecHit_);
  tok_EE_ = consumes<EcalRecHitCollection>(labelEERecHit_);
  tok_HBHE_ = consumes<HBHERecHitCollection>(labelHBHERecHit_);
  tok_Vtx_ = consumes<reco::VertexCollection>(labelVtx_);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  edm::LogVerbatim("HBHEMuon") << "Labels used: Track " << labelGenTrack_ << " Vtx " << labelVtx_ << " EB "
                               << labelEBRecHit_ << " EE " << labelEERecHit_ << " HBHE " << labelHBHERecHit_ << " MU "
                               << labelMuon_;

  if (!fileInCorr_.empty()) {
    std::ifstream infile(fileInCorr_.c_str());
    if (infile.is_open()) {
      while (true) {
        unsigned int id;
        double cfac;
        infile >> id >> cfac;
        if (!infile.good())
          break;
        corrValue_[DetId(id)] = cfac;
      }
      infile.close();
    }
  }
  useMyCorr_ = (!corrValue_.empty());
  edm::LogVerbatim("HBHEMuon") << "Flags used: UseRaw " << useRaw_ << " GetCharge " << getCharge_ << " UnCorrect "
                               << unCorrect_ << " IgnoreHECorr " << ignoreHECorr_ << " CollapseDepth " << collapseDepth_
                               << ":" << mergedDepth_ << " IsItPlan1 " << isItPlan1_ << " IsItPreRecHit "
                               << isItPreRecHit_ << " UseMyCorr " << useMyCorr_;
}

//
// member functions
//

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalHBHEMuonHighEtaAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("labelEBRecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("labelEERecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("labelHBHERecHit", edm::InputTag("hbhereco"));
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelMuon", "muons");
  desc.add<std::string>("labelTrack", "generalTracks");
  desc.add<double>("etaMin", 2.0);
  desc.add<double>("emaxNearPThreshold", 10.0);
  desc.add<bool>("analyzeMuon", true);
  desc.add<bool>("unCorrect", false);
  desc.add<bool>("collapseDepth", false);
  desc.add<bool>("isItPlan1", false);
  desc.add<bool>("getCharge", false);
  desc.add<int>("useRaw", 0);
  desc.add<int>("verbosity", 0);
  desc.addUntracked<std::string>("fileInCorr", "");
  desc.addUntracked<std::string>("trackQuality", "highPurity");
  desc.addUntracked<double>("minTrackPt", 1.0);
  desc.addUntracked<double>("maxDxyPV", 0.02);
  desc.addUntracked<double>("maxDzPV", 100.0);
  desc.addUntracked<double>("maxChi2", 5.0);
  desc.addUntracked<double>("maxDpOverP", 0.1);
  desc.addUntracked<bool>("ignoreHECorr", false);
  desc.addUntracked<bool>("isItPreRecHit", false);
  desc.addUntracked<bool>("writeRespCorr", false);
  desc.addUntracked<int>("maxDepth", 7);
  descriptions.add("hcalHBHEMuonHighEta", desc);
}

// ------------ method called once each job just before starting event loop  ------------
void HcalHBHEMuonHighEtaAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("HBHEMuonHighEta", "HBHEMuonHighEta");
  tree_->Branch("pt_of_muon", &ptGlob_);
  tree_->Branch("eta_of_muon", &etaGlob_);
  tree_->Branch("phi_of_muon", &phiGlob_);
  tree_->Branch("energy_of_muon", &energyMuon_);
  tree_->Branch("p_of_muon", &pMuon_);
  tree_->Branch("MediumMuon", &mediumMuon_);
  tree_->Branch("IsolationR04", &isolationR04_);
  tree_->Branch("IsolationR03", &isolationR03_);
  tree_->Branch("ecal_3into3", &ecalEnergy_);
  tree_->Branch("hcal_3into3", &hcalEnergy_);
  tree_->Branch("ho_3into3", &hoEnergy_);
  tree_->Branch("emaxNearP", &emaxNearP_);

  tree_->Branch("Run_No", &runNumber_);
  tree_->Branch("Event_No", &eventNumber_);
  tree_->Branch("GoodVertex", &goodVertex_);
  tree_->Branch("matchedId", &matchedId_);
  tree_->Branch("hcal_cellHot", &hcalHot_);
  tree_->Branch("ecal_3x3", &ecal3x3Energy_);
  tree_->Branch("hcal_1x1", &hcal1x1Energy_);
  tree_->Branch("ecal_detID", &ecalDetId_);
  tree_->Branch("hcal_detID", &hcalDetId_);
  tree_->Branch("ehcal_detID", &ehcalDetId_);
  tree_->Branch("hcal_ieta", &hcal_ieta_);
  tree_->Branch("hcal_iphi", &hcal_iphi_);

  char name[100];
  for (int k = 0; k < maxDepth_; ++k) {
    sprintf(name, "hcal_edepth%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergy_[k]);
    sprintf(name, "hcal_activeL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLength_[k]);
    sprintf(name, "hcal_edepthHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyHot_[k]);
    sprintf(name, "hcal_activeHotL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLengthHot_[k]);
    sprintf(name, "hcal_cdepthHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthChargeHot_[k]);
    sprintf(name, "hcal_cdepthHotBG%d", (k + 1));
    tree_->Branch(name, &hcalDepthChargeHotBG_[k]);
    sprintf(name, "hcal_edepthCorrect%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyCorr_[k]);
    sprintf(name, "hcal_edepthHotCorrect%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyHotCorr_[k]);
    sprintf(name, "hcal_depthMatch%d", (k + 1));
    tree_->Branch(name, &hcalDepthMatch_[k]);
    sprintf(name, "hcal_depthMatchHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthMatchHot_[k]);
  }
  tree_->Branch("activeLength", &hcalActiveLength_);
  tree_->Branch("activeLengthHot", &hcalActiveLengthHot_);
  tree_->Branch("trackDz", &trackDz_);
  tree_->Branch("trackLayerCrossed", &trackLayerCrossed_);
  tree_->Branch("trackOuterHit", &trackOuterHit_);
  tree_->Branch("trackMissedInnerHits", &trackMissedInnerHits_);
  tree_->Branch("trackMissedOuterHits", &trackMissedOuterHits_);
}

// ------------ method called for each event  ------------
void HcalHBHEMuonHighEtaAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++kount_;
  clearVectors();
  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber_ << " Event " << eventNumber_;
#endif

  // get handles to calogeometry and calotopology
  iSetup.get<CaloGeometryRecord>().get(pG_);

  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH_);
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus_);
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv_);
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology_);
  iSetup.get<HcalDbRecord>().get(conditions_);

  // Relevant blocks from iEvent
  edm::Handle<reco::VertexCollection> vtx;
  iEvent.getByToken(tok_Vtx_, vtx);

  iEvent.getByToken(tok_EB_, barrelRecHitsHandle_);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle_);
  iEvent.getByToken(tok_HBHE_, hbhe_);

  // require a good vertex
  math::XYZPoint pvx;
  goodVertex_ = 0;
  if (!vtx.isValid()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "No Good Vertex found == Reject\n";
#endif
    return;
  }

  reco::VertexCollection::const_iterator firstGoodVertex = vtx->end();
  for (reco::VertexCollection::const_iterator it = vtx->begin(); it != vtx->end(); it++) {
    if (isGoodVertex(*it)) {
      if (firstGoodVertex == vtx->end())
        firstGoodVertex = it;
      ++goodVertex_;
    }
  }
  if (firstGoodVertex != vtx->end())
    pvx = firstGoodVertex->position();

  bool accept(false);
  if (barrelRecHitsHandle_.isValid() && endcapRecHitsHandle_.isValid() && hbhe_.isValid()) {
    accept = analyzeMuon_ ? analyzeMuon(iEvent, pvx) : analyzeHadron(iEvent, pvx);
  }
  if (accept) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "Total of " << hcal_ieta_.size() << " propagated points";
    for (unsigned int i = 0; i < hcal_ieta_.size(); ++i)
      edm::LogVerbatim("HBHEMuon") << "[" << i << "] ieta/iphi for entry to "
                                   << "HCAL has value of " << hcal_ieta_[i] << ":" << hcal_iphi_[i];
    if ((verbosity_ / 100) % 10 > 0) {
      edm::LogVerbatim("HBHEMuon") << "Sizes:: ptGlob:" << ptGlob_.size() << " etaGlob:" << etaGlob_.size()
                                   << " phiGlob:" << phiGlob_.size() << " energyMuon:" << energyMuon_.size()
                                   << " pMuon:" << pMuon_.size() << " mediumMuon: " << mediumMuon_.size()
                                   << " isolation:" << isolationR04_.size() << ":" << isolationR03_.size()
                                   << " e|h|ho energy: " << ecalEnergy_.size() << ":" << hcalEnergy_.size() << ":"
                                   << hoEnergy_.size();
      edm::LogVerbatim("HBHEMuon") << "        matchedId:" << matchedId_.size() << " hcalHot:" << hcalHot_.size()
                                   << " 3x3|1x1 energy:" << ecal3x3Energy_.size() << ":" << hcal1x1Energy_.size()
                                   << " detId:" << ecalDetId_.size() << ":" << hcalDetId_.size() << ":"
                                   << ehcalDetId_.size() << " eta|phi:" << hcal_ieta_.size() << ":"
                                   << hcal_iphi_.size();
      edm::LogVerbatim("HBHEMuon") << "        activeLength:" << hcalActiveLength_.size() << ":"
                                   << hcalActiveLengthHot_.size() << " emaxNearP:" << emaxNearP_.size()
                                   << " trackDz: " << trackDz_.size() << " tracks:" << trackLayerCrossed_.size() << ":"
                                   << trackOuterHit_.size() << ":" << trackMissedInnerHits_.size() << ":"
                                   << trackMissedOuterHits_.size();
      for (unsigned int i = 0; i < depthMax_; ++i)
        edm::LogVerbatim("HBHEMuon")
            << "Depth " << i
            << " Energy|Length|EnergyHot|LengthHot|Charge|ChargeBG|EnergyCorr|EnergyHotCorr|Match|MatchHot:"
            << hcalDepthEnergy_[i].size() << ":" << hcalDepthActiveLength_[i].size() << ":"
            << hcalDepthEnergyHot_[i].size() << ":" << hcalDepthActiveLengthHot_[i].size() << ":"
            << hcalDepthChargeHot_[i].size() << ":" << hcalDepthChargeHotBG_[i].size() << ":"
            << hcalDepthEnergyCorr_[i].size() << ":" << hcalDepthEnergyHotCorr_[i].size() << ":"
            << hcalDepthMatch_[i].size() << ":" << hcalDepthMatchHot_[i].size();
    }
#endif
    tree_->Fill();
  }
}

// ------------ method called when starting to processes a run  ------------
void HcalHBHEMuonHighEtaAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  hdc_ = pHRNDC.product();
  actHB.clear();
  actHE.clear();
  actHB = hdc_->getThickActive(0);
  actHE = hdc_->getThickActive(1);
#ifdef EDM_ML_DEBUG
  if (verbosity_ % 10 > 0) {
    unsigned int k1(0), k2(0);
    edm::LogVerbatim("HBHEMuon") << actHB.size() << " Active Length for HB";
    for (const auto& act : actHB) {
      edm::LogVerbatim("HBHEMuon") << "[" << k1 << "] ieta " << act.ieta << " depth " << act.depth << " zside "
                                   << act.zside << " type " << act.stype << " phi " << act.iphis.size() << ":"
                                   << act.iphis[0] << " L " << act.thick;
      HcalDetId hcid1(HcalBarrel, (act.ieta) * (act.zside), act.iphis[0], act.depth);
      HcalDetId hcid2 = mergedDepth_ ? hdc_->mergedDepthDetId(hcid1) : hcid1;
      edm::LogVerbatim("HBHEMuon") << hcid1 << " | " << hcid2 << " L " << activeLength(DetId(hcid2));
      ++k1;
    }
    edm::LogVerbatim("HBHEMuon") << actHE.size() << " Active Length for HE";
    for (const auto& act : actHE) {
      edm::LogVerbatim("HBHEMuon") << "[" << k2 << "] ieta " << act.ieta << " depth " << act.depth << " zside "
                                   << act.zside << " type " << act.stype << " phi " << act.iphis.size() << ":"
                                   << act.iphis[0] << " L " << act.thick;
      HcalDetId hcid1(HcalEndcap, (act.ieta) * (act.zside), act.iphis[0], act.depth);
      HcalDetId hcid2 = mergedDepth_ ? hdc_->mergedDepthDetId(hcid1) : hcid1;
      edm::LogVerbatim("HBHEMuon") << hcid1 << " | " << hcid2 << " L " << activeLength(DetId(hcid2));
      ++k2;
    }
  }
#endif

  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  theHBHETopology_ = htopo.product();

  edm::ESHandle<HcalRespCorrs> resp;
  iSetup.get<HcalRespCorrsRcd>().get(resp);
  respCorrs_ = new HcalRespCorrs(*resp.product());
  respCorrs_->setTopo(theHBHETopology_);

  // Write correction factors for all HB/HE events
  if (writeRespCorr_) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
    const HcalGeometry* gHcal = (const HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
    const std::vector<DetId>& ids = gHcal->getValidDetIds(DetId::Hcal, 0);
    edm::LogVerbatim("HBHEMuon") << "\nTable of Correction Factors for Run " << iRun.run() << "\n";
    for (auto const& id : ids) {
      if ((id.det() == DetId::Hcal) && ((id.subdetId() == HcalBarrel) || (id.subdetId() == HcalEndcap))) {
        edm::LogVerbatim("HBHEMuon") << HcalDetId(id) << " " << id.rawId() << " "
                                     << (respCorrs_->getValues(id))->getValue();
      }
    }
  }
}

bool HcalHBHEMuonHighEtaAnalyzer::analyzeMuon(const edm::Event& iEvent, math::XYZPoint& leadPV) {
  edm::Handle<reco::MuonCollection> _Muon;
  iEvent.getByToken(tok_Muon_, _Muon);
  bool accept = false;

  if (_Muon.isValid()) {
    int nTrack(0);
    std::vector<spr::propagatedTrackID> trkCaloDets;
    for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon != _Muon->end(); ++RecMuon) {
      if (RecMuon->innerTrack().isNonnull()) {
        const reco::Track* pTrack = (RecMuon->innerTrack()).get();
        if (std::abs(pTrack->eta()) > etaMin_) {
          if (analyzeTracks(pTrack, leadPV, nTrack, trkCaloDets, false)) {
            accept = true;
            ptGlob_.emplace_back((RecMuon)->pt());
            etaGlob_.emplace_back(RecMuon->eta());
            phiGlob_.emplace_back(RecMuon->phi());
            energyMuon_.push_back(RecMuon->energy());
            pMuon_.emplace_back(RecMuon->p());
            bool mediumMuon = (((RecMuon->isPFMuon()) && (RecMuon->isGlobalMuon() || RecMuon->isTrackerMuon())) &&
                               (RecMuon->innerTrack()->validFraction() > 0.49));
            if (mediumMuon) {
              double chiGlobal =
                  ((RecMuon->globalTrack().isNonnull()) ? RecMuon->globalTrack()->normalizedChi2() : 999);
              bool goodGlob =
                  (RecMuon->isGlobalMuon() && chiGlobal < 3 && RecMuon->combinedQuality().chi2LocalPosition < 12 &&
                   RecMuon->combinedQuality().trkKink < 20);
              mediumMuon = muon::segmentCompatibility(*RecMuon) > (goodGlob ? 0.303 : 0.451);
            }
            mediumMuon_.emplace_back(mediumMuon);
            bool isoR03 =
                ((RecMuon->pfIsolationR03().sumChargedHadronPt +
                  std::max(0.,
                           RecMuon->pfIsolationR03().sumNeutralHadronEt + RecMuon->pfIsolationR03().sumPhotonEt -
                               (0.5 * RecMuon->pfIsolationR03().sumPUPt))) /
                 RecMuon->pt());
            bool isoR04 =
                ((RecMuon->pfIsolationR04().sumChargedHadronPt +
                  std::max(0.,
                           RecMuon->pfIsolationR04().sumNeutralHadronEt + RecMuon->pfIsolationR04().sumPhotonEt -
                               (0.5 * RecMuon->pfIsolationR04().sumPUPt))) /
                 RecMuon->pt());
            isolationR03_.emplace_back(isoR03);
            isolationR04_.emplace_back(isoR04);

            ecalEnergy_.emplace_back(RecMuon->calEnergy().emS9);
            hcalEnergy_.emplace_back(RecMuon->calEnergy().hadS9);
            hoEnergy_.emplace_back(RecMuon->calEnergy().hoS9);
#ifdef EDM_ML_DEBUG
            if ((verbosity_ / 100) % 10 > 0)
              edm::LogVerbatim("HBHEMuon")
                  << "Muon[" << ptGlob_.size() << "] pt:eta:phi:p " << ptGlob_.back() << ":" << etaGlob_.back() << ":"
                  << phiGlob_.back() << ":" << energyMuon_.back() << ":" << pMuon_.back() << ":"
                  << " Medium:i3:i4 " << mediumMuon_.back() << ":" << isolationR03_.back() << ":"
                  << isolationR04_.back() << ":"
                  << " Energy EC:HC:HO " << ecalEnergy_.back() << ":" << hcalEnergy_.back() << ":" << hoEnergy_.back();
#endif
          }
        }
      }
    }
  }
  return accept;
}

bool HcalHBHEMuonHighEtaAnalyzer::analyzeHadron(const edm::Event& iEvent, math::XYZPoint& leadPV) {
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  bool accept = false;

  if (!trkCollection.isValid()) {
    const CaloGeometry* geo = pG_.product();
    const MagneticField* bField = bFieldH_.product();
    std::vector<spr::propagatedTrackID> trkCaloDets;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDets, false);
    int nTrack(0);
    std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;
    for (trkDetItr = trkCaloDets.begin(), nTrack = 0; trkDetItr != trkCaloDets.end(); trkDetItr++, nTrack++) {
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      if (std::abs(pTrack->eta()) > etaMin_) {
        accept = analyzeTracks(pTrack, leadPV, nTrack, trkCaloDets, true);
      }
    }
  }
  return accept;
}

bool HcalHBHEMuonHighEtaAnalyzer::analyzeTracks(const reco::Track* pTrack,
                                                math::XYZPoint& leadPV,
                                                int nTrack,
                                                std::vector<spr::propagatedTrackID>& trkCaloDets,
                                                bool ifHadron) {
  const CaloGeometry* geo = pG_.product();
  const MagneticField* bField = bFieldH_.product();
  const EcalChannelStatus* theEcalChStatus = ecalChStatus_.product();
  const CaloTopology* caloTopology = theCaloTopology_.product();
  bool accept(false);

  if (spr::goodTrack(pTrack, leadPV, selectionParameter_, false)) {
    spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, false);

    if (trackID.okECAL && trackID.okHCAL) {
      double emaxNearP = (ifHadron) ? spr::chargeIsolationEcal(nTrack, trkCaloDets, geo, caloTopology, 15, 15) : 0;
      if (emaxNearP < emaxNearPThr_) {
        double eEcal(0), eHcal(0), activeLengthTot(0), activeLengthHotTot(0);
        double eHcalDepth[depthMax_], eHcalDepthHot[depthMax_];
        double eHcalDepthC[depthMax_], eHcalDepthHotC[depthMax_];
        double cHcalDepthHot[depthMax_], cHcalDepthHotBG[depthMax_];
        double activeL[depthMax_], activeHotL[depthMax_];
        bool matchDepth[depthMax_], matchDepthHot[depthMax_];
        HcalDetId eHcalDetId[depthMax_];
        unsigned int isHot(0);
        bool tmpmatch(false);
        int ieta(-1000), iphi(-1000);
        for (int i = 0; i < depthMax_; ++i) {
          eHcalDepth[i] = eHcalDepthHot[i] = 0;
          eHcalDepthC[i] = eHcalDepthHotC[i] = 0;
          cHcalDepthHot[i] = cHcalDepthHotBG[i] = 0;
          activeL[i] = activeHotL[i] = 0;
          matchDepth[i] = matchDepthHot[i] = true;
        }

        HcalDetId check;
        std::pair<bool, HcalDetId> info = spr::propagateHCALBack(pTrack, geo, bField, false);
        if (info.first)
          check = info.second;

        const DetId isoCell(trackID.detIdECAL);
        std::pair<double, bool> e3x3 = spr::eECALmatrix(isoCell,
                                                        barrelRecHitsHandle_,
                                                        endcapRecHitsHandle_,
                                                        *theEcalChStatus,
                                                        geo,
                                                        caloTopology,
                                                        sevlv_.product(),
                                                        1,
                                                        1,
                                                        -100.0,
                                                        -100.0,
                                                        -500.0,
                                                        500.0,
                                                        false);
        eEcal = e3x3.first;
#ifdef EDM_ML_DEBUG
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Propagate Track to ECAL: " << e3x3.second << ":" << trackID.okECAL << " E "
                                       << eEcal;
#endif

        DetId closestCell(trackID.detIdHCAL);
        HcalDetId hcidt(closestCell.rawId());
        if ((hcidt.ieta() == check.ieta()) && (hcidt.iphi() == check.iphi()))
          tmpmatch = true;
#ifdef EDM_ML_DEBUG
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Front " << hcidt << " Back " << info.first << ":" << check << " Match "
                                       << tmpmatch;
#endif

        HcalSubdetector subdet = hcidt.subdet();
        ieta = hcidt.ieta();
        iphi = hcidt.iphi();
        bool hborhe = (std::abs(ieta) == 16);

        eHcal = spr::eHCALmatrix(theHBHETopology_,
                                 closestCell,
                                 hbhe_,
                                 0,
                                 0,
                                 false,
                                 true,
                                 -100.0,
                                 -100.0,
                                 -100.0,
                                 -100.0,
                                 -500.,
                                 500.,
                                 useRaw_);
        std::vector<std::pair<double, int>> ehdepth;
        spr::energyHCALCell((HcalDetId)closestCell,
                            hbhe_,
                            ehdepth,
                            depthMax_,
                            -100.0,
                            -100.0,
                            -100.0,
                            -100.0,
                            -500.0,
                            500.0,
                            useRaw_,
                            depth16HE(ieta, iphi),
                            false);
        for (int i = 0; i < depthMax_; ++i)
          eHcalDetId[i] = HcalDetId();
        for (unsigned int i = 0; i < ehdepth.size(); ++i) {
          HcalSubdetector subdet0 =
              (hborhe) ? ((ehdepth[i].second >= depth16HE(ieta, iphi)) ? HcalEndcap : HcalBarrel) : subdet;
          HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
          double actL = activeLength(DetId(hcid0));
          double ene = ehdepth[i].first;
          bool tmpC(false);
          if (ene > 0.0) {
            if (!(theHBHETopology_->validHcal(hcid0))) {
              edm::LogWarning("HBHEMuon") << "(1) Invalid ID " << hcid0 << " with E = " << ene;
              edm::LogWarning("HBHEMuon") << HcalDetId(closestCell) << " with " << ehdepth.size() << " depths:";
              for (const auto& ehd : ehdepth)
                edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
            } else {
              tmpC = goodCell(hcid0, pTrack, geo, bField);
              double enec(ene);
              if (unCorrect_) {
                double corr = (ignoreHECorr_ && (subdet0 == HcalEndcap)) ? 1.0 : respCorr(DetId(hcid0));
                if (corr != 0)
                  ene /= corr;
#ifdef EDM_ML_DEBUG
                if (verbosity_ % 10 > 0) {
                  HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc_->mergedDepthDetId(hcid0) : hcid0;
                  edm::LogVerbatim("HBHEMuon") << hcid0 << ":" << id << " Corr " << corr;
                }
#endif
              }
              int depth = ehdepth[i].second - 1;
              if (collapseDepth_) {
                HcalDetId id = hdc_->mergedDepthDetId(hcid0);
                depth = id.depth() - 1;
              }
              eHcalDepth[depth] += ene;
              eHcalDepthC[depth] += enec;
              activeL[depth] += actL;
              activeLengthTot += actL;
              matchDepth[depth] = (matchDepth[depth] && tmpC);
#ifdef EDM_ML_DEBUG
              if ((verbosity_ / 10) % 10 > 0)
                edm::LogVerbatim("HBHEMuon")
                    << hcid0 << " E " << ene << ":" << enec << " L " << actL << " Match " << tmpC;
#endif
            }
          }
        }
#ifdef EDM_ML_DEBUG
        if ((verbosity_ / 10) % 10 > 0) {
          edm::LogVerbatim("HBHEMuon") << hcidt << " Match " << tmpmatch << " Depths " << ehdepth.size();
          for (unsigned int k = 0; k < ehdepth.size(); ++k)
            edm::LogVerbatim("HBHEMuon") << " [" << k << ":" << ehdepth[k].second << "] " << matchDepth[k];
        }
#endif
        HcalDetId hotCell;
        spr::eHCALmatrix(geo, theHBHETopology_, closestCell, hbhe_, 1, 1, hotCell, false, useRaw_, false);
        isHot = matchId(closestCell, hotCell);
        if (hotCell != HcalDetId()) {
          subdet = HcalDetId(hotCell).subdet();
          ieta = HcalDetId(hotCell).ieta();
          iphi = HcalDetId(hotCell).iphi();
          hborhe = (std::abs(ieta) == 16);
          std::vector<std::pair<double, int>> ehdepth;
          spr::energyHCALCell(hotCell,
                              hbhe_,
                              ehdepth,
                              depthMax_,
                              -100.0,
                              -100.0,
                              -100.0,
                              -100.0,
                              -500.0,
                              500.0,
                              useRaw_,
                              depth16HE(ieta, iphi),
                              false);
          for (int i = 0; i < depthMax_; ++i)
            eHcalDetId[i] = HcalDetId();
          for (unsigned int i = 0; i < ehdepth.size(); ++i) {
            HcalSubdetector subdet0 =
                (hborhe) ? ((ehdepth[i].second >= depth16HE(ieta, iphi)) ? HcalEndcap : HcalBarrel) : subdet;
            HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
            double actL = activeLength(DetId(hcid0));
            double ene = ehdepth[i].first;
            bool tmpC(false);
            if (ene > 0.0) {
              if (!(theHBHETopology_->validHcal(hcid0))) {
                edm::LogWarning("HBHEMuon") << "(2) Invalid ID " << hcid0 << " with E = " << ene;
                edm::LogWarning("HBHEMuon") << HcalDetId(hotCell) << " with " << ehdepth.size() << " depths:";
                for (const auto& ehd : ehdepth)
                  edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
              } else {
                tmpC = goodCell(hcid0, pTrack, geo, bField);
                double chg(ene), enec(ene);
                if (unCorrect_) {
                  double corr = (ignoreHECorr_ && (subdet0 == HcalEndcap)) ? 1.0 : respCorr(DetId(hcid0));
                  if (corr != 0)
                    ene /= corr;
#ifdef EDM_ML_DEBUG
                  if (verbosity_ % 10 > 0) {
                    HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc_->mergedDepthDetId(hcid0) : hcid0;
                    edm::LogVerbatim("HBHEMuon")
                        << hcid0 << ":" << id << " Corr " << corr << " E " << ene << ":" << enec;
                  }
#endif
                }
                if (getCharge_) {
                  double gain = gainFactor(conditions_, hcid0);
                  if (gain != 0)
                    chg /= gain;
#ifdef EDM_ML_DEBUG
                  if (verbosity_ % 10 > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Gain " << gain << " C " << chg;
#endif
                }
                int depth = ehdepth[i].second - 1;
                if (collapseDepth_) {
                  HcalDetId id = hdc_->mergedDepthDetId(hcid0);
                  depth = id.depth() - 1;
                }
                eHcalDepthHot[depth] += ene;
                eHcalDepthHotC[depth] += enec;
                cHcalDepthHot[depth] += chg;
                activeHotL[depth] += actL;
                activeLengthHotTot += actL;
                matchDepthHot[depth] = (matchDepthHot[depth] && tmpC);
#ifdef EDM_ML_DEBUG
                if ((verbosity_ / 10) % 10 > 0)
                  edm::LogVerbatim("HBHEMuon") << hcid0 << " depth " << depth << " E " << ene << ":" << enec << " C "
                                               << chg << " L " << actL << " Match " << tmpC;
#endif
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HBHEMuon") << "Propagate Track to HCAL: " << trackID.okHCAL << " Match " << tmpmatch
                                     << " Hot " << isHot << " Energy " << eHcal;
#endif

        accept = true;
        ecalDetId_.emplace_back((trackID.detIdECAL)());
        hcalDetId_.emplace_back((trackID.detIdHCAL)());
        ehcalDetId_.emplace_back((trackID.detIdEHCAL)());
        emaxNearP_.emplace_back(emaxNearP);
        matchedId_.emplace_back(tmpmatch);
        ecal3x3Energy_.emplace_back(eEcal);
        hcal1x1Energy_.emplace_back(eHcal);
        hcal_ieta_.emplace_back(ieta);
        hcal_iphi_.emplace_back(iphi);
        for (int i = 0; i < maxDepth_; ++i) {
          hcalDepthEnergy_[i].emplace_back(eHcalDepth[i]);
          hcalDepthActiveLength_[i].emplace_back(activeL[i]);
          hcalDepthEnergyHot_[i].emplace_back(eHcalDepthHot[i]);
          hcalDepthActiveLengthHot_[i].emplace_back(activeHotL[i]);
          hcalDepthEnergyCorr_[i].emplace_back(eHcalDepthC[i]);
          hcalDepthEnergyHotCorr_[i].emplace_back(eHcalDepthHotC[i]);
          hcalDepthChargeHot_[i].emplace_back(cHcalDepthHot[i]);
          hcalDepthChargeHotBG_[i].emplace_back(cHcalDepthHotBG[i]);
          hcalDepthMatch_[i].emplace_back(matchDepth[i]);
          hcalDepthMatchHot_[i].emplace_back(matchDepthHot[i]);
        }
        hcalActiveLength_.emplace_back(activeLengthTot);
        hcalHot_.emplace_back(isHot);
        hcalActiveLengthHot_.emplace_back(activeLengthHotTot);
#ifdef EDM_ML_DEBUG
        if ((verbosity_ / 100) % 10 > 0) {
          edm::LogVerbatim("HBHEMuon") << "Track " << std::hex << ecalDetId_.back() << ":" << hcalDetId_.back() << ":"
                                       << ehcalDetId_.back() << std::dec << ":" << emaxNearP_.back() << ":"
                                       << matchedId_.back() << ":" << ecal3x3Energy_.back() << ":"
                                       << hcal1x1Energy_.back() << ":" << hcal_ieta_.back() << ":" << hcal_iphi_.back()
                                       << ":" << hcalActiveLength_.back() << ":" << hcalHot_.back() << ":"
                                       << hcalActiveLengthHot_.back();
          for (int i = 0; i < maxDepth_; ++i) {
            edm::LogVerbatim("HBHEMuon") << "Depth[" << i << "] " << hcalDepthEnergy_[i].back() << ":"
                                         << hcalDepthActiveLength_[i].back() << ":" << hcalDepthEnergyHot_[i].back()
                                         << ":" << hcalDepthActiveLengthHot_[i].back() << ":"
                                         << hcalDepthEnergyCorr_[i].back() << ":" << hcalDepthEnergyHotCorr_[i].back()
                                         << ":" << hcalDepthChargeHot_[i].back() << ":"
                                         << hcalDepthChargeHotBG_[i].back() << ":" << hcalDepthMatch_[i].back() << ":"
                                         << hcalDepthMatchHot_[i].back();
          }
        }
#endif
        fillTrackParameters(pTrack, leadPV);
      }
    }
  }
  return accept;
}

void HcalHBHEMuonHighEtaAnalyzer::clearVectors() {
  ///clearing vectots
  eventNumber_ = -99999;
  runNumber_ = -99999;
  goodVertex_ = -99999;

  mediumMuon_.clear();
  ptGlob_.clear();
  etaGlob_.clear();
  phiGlob_.clear();
  energyMuon_.clear();
  pMuon_.clear();
  isolationR04_.clear();
  isolationR03_.clear();
  ecalEnergy_.clear();
  hcalEnergy_.clear();
  hoEnergy_.clear();

  matchedId_.clear();
  hcalHot_.clear();
  ecal3x3Energy_.clear();
  hcal1x1Energy_.clear();
  ecalDetId_.clear();
  hcalDetId_.clear();
  ehcalDetId_.clear();
  hcal_ieta_.clear();
  hcal_iphi_.clear();
  for (int i = 0; i < depthMax_; ++i) {
    hcalDepthEnergy_[i].clear();
    hcalDepthActiveLength_[i].clear();
    hcalDepthEnergyHot_[i].clear();
    hcalDepthActiveLengthHot_[i].clear();
    hcalDepthChargeHot_[i].clear();
    hcalDepthChargeHotBG_[i].clear();
    hcalDepthEnergyCorr_[i].clear();
    hcalDepthEnergyHotCorr_[i].clear();
    hcalDepthMatch_[i].clear();
    hcalDepthMatchHot_[i].clear();
  }
  hcalActiveLength_.clear();
  hcalActiveLengthHot_.clear();

  emaxNearP_.clear();
  trackDz_.clear();
  trackLayerCrossed_.clear();
  trackOuterHit_.clear();
  trackMissedInnerHits_.clear();
  trackMissedOuterHits_.clear();
}

int HcalHBHEMuonHighEtaAnalyzer::matchId(const HcalDetId& id1, const HcalDetId& id2) {
  HcalDetId kd1(id1.subdet(), id1.ieta(), id1.iphi(), 1);
  HcalDetId kd2(id1.subdet(), id2.ieta(), id2.iphi(), 1);
  int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalHBHEMuonHighEtaAnalyzer::activeLength(const DetId& hid) {
  HcalDetId id(hid);
  int ieta = id.ietaAbs();
  int zside = id.zside();
  int iphi = id.iphi();
  std::vector<int> dpths;
  if (mergedDepth_) {
    std::vector<HcalDetId> ids;
    hdc_->unmergeDepthDetId(id, ids);
    for (auto idh : ids)
      dpths.emplace_back(idh.depth());
  } else {
    dpths.emplace_back(id.depth());
  }
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    for (unsigned int i = 0; i < actHB.size(); ++i) {
      if ((ieta == actHB[i].ieta) && (zside == actHB[i].zside) &&
          (std::find(dpths.begin(), dpths.end(), actHB[i].depth) != dpths.end()) &&
          (std::find(actHB[i].iphis.begin(), actHB[i].iphis.end(), iphi) != actHB[i].iphis.end())) {
        lx += actHB[i].thick;
      }
    }
  } else {
    for (unsigned int i = 0; i < actHE.size(); ++i) {
      if ((ieta == actHE[i].ieta) && (zside == actHE[i].zside) &&
          (std::find(dpths.begin(), dpths.end(), actHE[i].depth) != dpths.end()) &&
          (std::find(actHE[i].iphis.begin(), actHE[i].iphis.end(), iphi) != actHE[i].iphis.end())) {
        lx += actHE[i].thick;
      }
    }
  }
  return lx;
}

bool HcalHBHEMuonHighEtaAnalyzer::isGoodVertex(const reco::Vertex& vtx) {
  if (vtx.isFake())
    return false;
  if (vtx.ndof() < 4)
    return false;
  if (vtx.position().Rho() > 2.)
    return false;
  if (fabs(vtx.position().Z()) > 24.)
    return false;
  return true;
}

double HcalHBHEMuonHighEtaAnalyzer::respCorr(const DetId& id) {
  double cfac(1.0);
  if (useMyCorr_) {
    auto itr = corrValue_.find(id);
    if (itr != corrValue_.end())
      cfac = itr->second;
  } else if (respCorrs_ != nullptr) {
    cfac = (respCorrs_->getValues(id))->getValue();
  }
  return cfac;
}

double HcalHBHEMuonHighEtaAnalyzer::gainFactor(const edm::ESHandle<HcalDbService>& conditions, const HcalDetId& id) {
  double gain(0.0);
  const HcalCalibrations& calibs = conditions->getHcalCalibrations(id);
  for (int capid = 0; capid < 4; ++capid)
    gain += (0.25 * calibs.respcorrgain(capid));
  return gain;
}

int HcalHBHEMuonHighEtaAnalyzer::depth16HE(int ieta, int iphi) {
  // Transition between HB/HE is special
  // For Run 1 or for Plan1 standard reconstruction it is 3
  // For runs beyond 2018 or in Plan1 for HEP17 it is 4
  int zside = (ieta > 0) ? 1 : -1;
  int depth = theHBHETopology_->dddConstants()->getMinDepth(1, 16, iphi, zside);
  if (isItPlan1_ && (!isItPreRecHit_))
    depth = 3;
#ifdef EDM_ML_DEBUG
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("HBHEMuon") << "Plan1 " << isItPlan1_ << " PreRecHit " << isItPreRecHit_ << " phi " << iphi
                                 << " depth " << depth;
#endif
  return depth;
}

bool HcalHBHEMuonHighEtaAnalyzer::goodCell(const HcalDetId& hcid,
                                           const reco::Track* pTrack,
                                           const CaloGeometry* geo,
                                           const MagneticField* bField) {
  std::pair<double, double> rz = hdc_->getRZ(hcid);
  bool typeRZ = (hcid.subdet() == HcalEndcap) ? false : true;
  bool match = spr::propagateHCAL(pTrack, geo, bField, typeRZ, rz, false);
  return match;
}

void HcalHBHEMuonHighEtaAnalyzer::fillTrackParameters(const reco::Track* pTrack, math::XYZPoint leadPV) {
  trackDz_.emplace_back(pTrack->dz(leadPV));
  const reco::HitPattern& hitp = pTrack->hitPattern();
  trackLayerCrossed_.emplace_back(hitp.trackerLayersWithMeasurement());
  trackOuterHit_.emplace_back(hitp.stripTOBLayersWithMeasurement() + hitp.stripTECLayersWithMeasurement());
  trackMissedInnerHits_.emplace_back(hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));
  trackMissedOuterHits_.emplace_back(hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalHBHEMuonHighEtaAnalyzer);
