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
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#define EDM_ML_DEBUG

class HcalHBHEMuonAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalHBHEMuonAnalyzer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void clearVectors();
  int matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&);
  bool isGoodVertex(const reco::Vertex& vtx);
  double respCorr(const DetId& id);
  double gainFactor(const HcalDbService* dbserv, const HcalDetId& id);
  int depth16HE(int ieta, int iphi);
  bool goodCell(const HcalDetId& hcid, const reco::Track* pTrack, const CaloGeometry* geo, const MagneticField* bField);

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  const edm::InputTag hlTriggerResults_;
  const edm::InputTag labelEBRecHit_, labelEERecHit_, labelHBHERecHit_;
  const std::string labelVtx_, labelMuon_, fileInCorr_;
  const std::vector<std::string> triggers_;
  const double pMinMuon_;
  const int verbosity_, useRaw_;
  const bool unCorrect_, collapseDepth_, isItPlan1_;
  const bool ignoreHECorr_, isItPreRecHit_;
  const bool getCharge_, writeRespCorr_;
  const int maxDepth_;
  const std::string modnam_, procnm_;

  const edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  const edm::EDGetTokenT<reco::VertexCollection> tok_Vtx_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  const edm::EDGetTokenT<HBHERecHitCollection> tok_HBHE_;
  const edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;

  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respcorr_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_chan_;
  const edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_topo_;
  const edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbservice_;

  const HcalDDDRecConstants* hdc_;
  const HcalTopology* theHBHETopology_;
  const CaloGeometry* geo_;
  HcalRespCorrs* respCorrs_;

  bool mergedDepth_, useMyCorr_;
  int kount_;

  //////////////////////////////////////////////////////
  static const int depthMax_ = 7;
  TTree* tree_;
  unsigned int runNumber_, eventNumber_, lumiNumber_, bxNumber_;
  unsigned int goodVertex_;
  bool muon_is_good_, muon_global_, muon_tracker_;
  bool muon_is_tight_, muon_is_medium_;
  double ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  float muon_trkKink_, muon_chi2LocalPosition_, muon_segComp_;
  int trackerLayer_, numPixelLayers_, tight_PixelHits_;
  bool innerTrack_, outerTrack_, globalTrack_;
  double chiTracker_, dxyTracker_, dzTracker_;
  double innerTrackpt_, innerTracketa_, innerTrackphi_;
  double tight_validFraction_, outerTrackChi_;
  double outerTrackPt_, outerTrackEta_, outerTrackPhi_;
  int outerTrackHits_, outerTrackRHits_;
  double globalTrckPt_, globalTrckEta_, globalTrckPhi_;
  int globalMuonHits_, matchedStat_;
  double chiGlobal_, tight_LongPara_, tight_TransImpara_;
  double isolationR04_, isolationR03_;
  double ecalEnergy_, hcalEnergy_, hoEnergy_;
  bool matchedId_, hcalHot_;
  double ecal3x3Energy_, hcal1x1Energy_;
  unsigned int ecalDetId_, hcalDetId_, ehcalDetId_;
  int hcal_ieta_, hcal_iphi_;
  double hcalDepthEnergy_[depthMax_];
  double hcalDepthActiveLength_[depthMax_];
  double hcalDepthEnergyHot_[depthMax_];
  double hcalDepthActiveLengthHot_[depthMax_];
  double hcalDepthChargeHot_[depthMax_];
  double hcalDepthChargeHotBG_[depthMax_];
  double hcalDepthEnergyCorr_[depthMax_];
  double hcalDepthEnergyHotCorr_[depthMax_];
  bool hcalDepthMatch_[depthMax_];
  bool hcalDepthMatchHot_[depthMax_];
  double hcalActiveLength_, hcalActiveLengthHot_;
  std::vector<std::string> all_triggers_;
  std::vector<int> hltresults_;

  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB, actHE;
  std::map<DetId, double> corrValue_;
  ////////////////////////////////////////////////////////////
};

HcalHBHEMuonAnalyzer::HcalHBHEMuonAnalyzer(const edm::ParameterSet& iConfig)
    : hlTriggerResults_(iConfig.getParameter<edm::InputTag>("hlTriggerResults")),
      labelEBRecHit_(iConfig.getParameter<edm::InputTag>("labelEBRecHit")),
      labelEERecHit_(iConfig.getParameter<edm::InputTag>("labelEERecHit")),
      labelHBHERecHit_(iConfig.getParameter<edm::InputTag>("labelHBHERecHit")),
      labelVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelMuon_(iConfig.getParameter<std::string>("labelMuon")),
      fileInCorr_(iConfig.getUntrackedParameter<std::string>("fileInCorr", "")),
      triggers_(iConfig.getParameter<std::vector<std::string>>("triggers")),
      pMinMuon_(iConfig.getParameter<double>("pMinMuon")),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      useRaw_(iConfig.getParameter<int>("useRaw")),
      unCorrect_(iConfig.getParameter<bool>("unCorrect")),
      collapseDepth_(iConfig.getParameter<bool>("collapseDepth")),
      isItPlan1_(iConfig.getParameter<bool>("isItPlan1")),
      ignoreHECorr_(iConfig.getUntrackedParameter<bool>("ignoreHECorr", false)),
      isItPreRecHit_(iConfig.getUntrackedParameter<bool>("isItPreRecHit", false)),
      getCharge_(iConfig.getParameter<bool>("getCharge")),
      writeRespCorr_(iConfig.getUntrackedParameter<bool>("writeRespCorr", false)),
      maxDepth_(iConfig.getUntrackedParameter<int>("maxDepth", 7)),
      modnam_(iConfig.getUntrackedParameter<std::string>("moduleName", "")),
      procnm_(iConfig.getUntrackedParameter<std::string>("processName", "")),
      tok_trigRes_(consumes<edm::TriggerResults>(hlTriggerResults_)),
      tok_Vtx_((modnam_.empty()) ? consumes<reco::VertexCollection>(labelVtx_)
                                 : consumes<reco::VertexCollection>(edm::InputTag(modnam_, labelVtx_, procnm_))),
      tok_EB_(consumes<EcalRecHitCollection>(labelEBRecHit_)),
      tok_EE_(consumes<EcalRecHitCollection>(labelEERecHit_)),
      tok_HBHE_(consumes<HBHERecHitCollection>(labelHBHERecHit_)),
      tok_Muon_((modnam_.empty()) ? consumes<reco::MuonCollection>(labelMuon_)
                                  : consumes<reco::MuonCollection>(edm::InputTag(modnam_, labelMuon_, procnm_))),
      tok_ddrec_(esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      tok_htopo_(esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      tok_respcorr_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd, edm::Transition::BeginRun>()),
      tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      tok_magField_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      tok_chan_(esConsumes<EcalChannelStatus, EcalChannelStatusRcd>()),
      tok_sevlv_(esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>()),
      tok_topo_(esConsumes<CaloTopology, CaloTopologyRecord>()),
      tok_dbservice_(esConsumes<HcalDbService, HcalDbRecord>()),
      hdc_(nullptr),
      theHBHETopology_(nullptr),
      respCorrs_(nullptr) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  kount_ = 0;
  mergedDepth_ = (!isItPreRecHit_) || (collapseDepth_);

  if (modnam_.empty()) {
    edm::LogVerbatim("HBHEMuon") << "Labels used: Trig " << hlTriggerResults_ << " Vtx " << labelVtx_ << " EB "
                                 << labelEBRecHit_ << " EE " << labelEERecHit_ << " HBHE " << labelHBHERecHit_ << " MU "
                                 << labelMuon_;
  } else {
    edm::LogVerbatim("HBHEMuon") << "Labels used Trig " << hlTriggerResults_ << "\n  Vtx  "
                                 << edm::InputTag(modnam_, labelVtx_, procnm_) << "\n  EB   " << labelEBRecHit_
                                 << "\n  EE   " << labelEERecHit_ << "\n  HBHE " << labelHBHERecHit_ << "\n  MU   "
                                 << edm::InputTag(modnam_, labelMuon_, procnm_);
  }

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
                               << isItPreRecHit_ << " UseMyCorr " << useMyCorr_ << " pMinMuon " << pMinMuon_;
}

//
// member functions
//

// ------------ method called for each event  ------------
void HcalHBHEMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++kount_;
  clearVectors();
  std::vector<bool> muon_is_good, muon_global, muon_tracker;
  std::vector<bool> muon_is_tight, muon_is_medium;
  std::vector<double> ptGlob, etaGlob, phiGlob, energyMuon, pMuon;
  std::vector<float> muon_trkKink, muon_chi2LocalPosition, muon_segComp;
  std::vector<int> trackerLayer, numPixelLayers, tight_PixelHits;
  std::vector<bool> innerTrack, outerTrack, globalTrack;
  std::vector<double> chiTracker, dxyTracker, dzTracker;
  std::vector<double> innerTrackpt, innerTracketa, innerTrackphi;
  std::vector<double> tight_validFraction, outerTrackChi;
  std::vector<double> outerTrackPt, outerTrackEta, outerTrackPhi;
  std::vector<int> outerTrackHits, outerTrackRHits;
  std::vector<double> globalTrckPt, globalTrckEta, globalTrckPhi;
  std::vector<int> globalMuonHits, matchedStat;
  std::vector<double> chiGlobal, tight_LongPara, tight_TransImpara;
  std::vector<double> isolationR04, isolationR03;
  std::vector<double> ecalEnergy, hcalEnergy, hoEnergy;
  std::vector<bool> matchedId, hcalHot;
  std::vector<double> ecal3x3Energy, hcal1x1Energy;
  std::vector<unsigned int> ecalDetId, hcalDetId, ehcalDetId;
  std::vector<int> hcal_ieta, hcal_iphi;
  std::vector<double> hcalDepthEnergy[depthMax_];
  std::vector<double> hcalDepthActiveLength[depthMax_];
  std::vector<double> hcalDepthEnergyHot[depthMax_];
  std::vector<double> hcalDepthActiveLengthHot[depthMax_];
  std::vector<double> hcalDepthChargeHot[depthMax_];
  std::vector<double> hcalDepthChargeHotBG[depthMax_];
  std::vector<double> hcalDepthEnergyCorr[depthMax_];
  std::vector<double> hcalDepthEnergyHotCorr[depthMax_];
  std::vector<bool> hcalDepthMatch[depthMax_];
  std::vector<bool> hcalDepthMatchHot[depthMax_];
  std::vector<double> hcalActiveLength, hcalActiveLengthHot;
  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_ = iEvent.id().luminosityBlock();
  bxNumber_ = iEvent.bunchCrossing();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber_ << " Event " << eventNumber_ << " Lumi " << lumiNumber_ << " BX "
                               << bxNumber_ << std::endl;
#endif
  const edm::Handle<edm::TriggerResults>& _Triggers = iEvent.getHandle(tok_trigRes_);
#ifdef EDM_ML_DEBUG
  if ((verbosity_ / 10000) % 10 > 0)
    edm::LogVerbatim("HBHEMuon") << "Size of all triggers " << all_triggers_.size();
#endif
  int Ntriggers = static_cast<int>(all_triggers_.size());
#ifdef EDM_ML_DEBUG
  if ((verbosity_ / 10000) % 10 > 0)
    edm::LogVerbatim("HBHEMuon") << "Size of HLT MENU: " << _Triggers->size();
#endif
  if (_Triggers.isValid()) {
    const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*_Triggers);
    std::vector<int> index;
    for (int i = 0; i < Ntriggers; i++) {
      index.push_back(triggerNames_.triggerIndex(all_triggers_[i]));
      int triggerSize = static_cast<int>(_Triggers->size());
#ifdef EDM_ML_DEBUG
      if ((verbosity_ / 10000) % 10 > 0)
        edm::LogVerbatim("HBHEMuon") << "outside loop " << index[i] << "\ntriggerSize " << triggerSize;
#endif
      if (index[i] < triggerSize) {
        hltresults_.push_back(_Triggers->accept(index[i]));
#ifdef EDM_ML_DEBUG
        if ((verbosity_ / 10000) % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Trigger_info " << triggerSize << " triggerSize " << index[i]
                                       << " trigger_index " << hltresults_.at(i) << " hltresult";
#endif
      } else {
        if ((verbosity_ / 10000) % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Requested HLT path \""
                                       << "\" does not exist";
      }
    }
  }

  // get handles to calogeometry and calotopology
  const MagneticField* bField = &iSetup.getData(tok_magField_);
  const EcalChannelStatus* theEcalChStatus = &iSetup.getData(tok_chan_);
  const EcalSeverityLevelAlgo* sevlv = &iSetup.getData(tok_sevlv_);
  const CaloTopology* caloTopology = &iSetup.getData(tok_topo_);
  const HcalDbService* conditions = &iSetup.getData(tok_dbservice_);

  // Relevant blocks from iEvent
  const edm::Handle<reco::VertexCollection>& vtx = iEvent.getHandle(tok_Vtx_);

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle = iEvent.getHandle(tok_EB_);
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle = iEvent.getHandle(tok_EE_);

  edm::Handle<HBHERecHitCollection> hbhe = iEvent.getHandle(tok_HBHE_);

  const edm::Handle<reco::MuonCollection>& _Muon = iEvent.getHandle(tok_Muon_);

  // require a good vertex
  math::XYZPoint pvx;
  goodVertex_ = 0;
  if (!vtx.isValid()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "No Good Vertex found == Reject";
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
  if (_Muon.isValid() && barrelRecHitsHandle.isValid() && endcapRecHitsHandle.isValid() && hbhe.isValid()) {
    for (const auto& RecMuon : (*(_Muon.product()))) {
      muon_is_good.push_back(RecMuon.isPFMuon());
      muon_global.push_back(RecMuon.isGlobalMuon());
      muon_tracker.push_back(RecMuon.isTrackerMuon());
      ptGlob.push_back(RecMuon.pt());
      etaGlob.push_back(RecMuon.eta());
      phiGlob.push_back(RecMuon.phi());
      energyMuon.push_back(RecMuon.energy());
      pMuon.push_back(RecMuon.p());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HBHEMuon") << "Energy:" << RecMuon.energy() << " P:" << RecMuon.p();
#endif
      muon_is_tight.push_back(muon::isTightMuon(RecMuon, *firstGoodVertex));
      muon_is_medium.push_back(muon::isMediumMuon(RecMuon));
      muon_trkKink.push_back(RecMuon.combinedQuality().trkKink);
      muon_chi2LocalPosition.push_back(RecMuon.combinedQuality().chi2LocalPosition);
      muon_segComp.push_back(muon::segmentCompatibility(RecMuon));
      // acessing tracker hits info
      if (RecMuon.track().isNonnull()) {
        trackerLayer.push_back(RecMuon.track()->hitPattern().trackerLayersWithMeasurement());
      } else {
        trackerLayer.push_back(-1);
      }
      if (RecMuon.innerTrack().isNonnull()) {
        innerTrack.push_back(true);
        numPixelLayers.push_back(RecMuon.innerTrack()->hitPattern().pixelLayersWithMeasurement());
        chiTracker.push_back(RecMuon.innerTrack()->normalizedChi2());
        dxyTracker.push_back(fabs(RecMuon.innerTrack()->dxy(pvx)));
        dzTracker.push_back(fabs(RecMuon.innerTrack()->dz(pvx)));
        innerTrackpt.push_back(RecMuon.innerTrack()->pt());
        innerTracketa.push_back(RecMuon.innerTrack()->eta());
        innerTrackphi.push_back(RecMuon.innerTrack()->phi());
        tight_PixelHits.push_back(RecMuon.innerTrack()->hitPattern().numberOfValidPixelHits());
        tight_validFraction.push_back(RecMuon.innerTrack()->validFraction());
      } else {
        innerTrack.push_back(false);
        numPixelLayers.push_back(0);
        chiTracker.push_back(0);
        dxyTracker.push_back(0);
        dzTracker.push_back(0);
        innerTrackpt.push_back(0);
        innerTracketa.push_back(0);
        innerTrackphi.push_back(0);
        tight_PixelHits.push_back(0);
        tight_validFraction.push_back(-99);
      }
      // outer track info
      if (RecMuon.outerTrack().isNonnull()) {
        outerTrack.push_back(true);
        outerTrackPt.push_back(RecMuon.outerTrack()->pt());
        outerTrackEta.push_back(RecMuon.outerTrack()->eta());
        outerTrackPhi.push_back(RecMuon.outerTrack()->phi());
        outerTrackChi.push_back(RecMuon.outerTrack()->normalizedChi2());
        outerTrackHits.push_back(RecMuon.outerTrack()->numberOfValidHits());
        outerTrackRHits.push_back(RecMuon.outerTrack()->recHitsSize());
      } else {
        outerTrack.push_back(false);
        outerTrackPt.push_back(0);
        outerTrackEta.push_back(0);
        outerTrackPhi.push_back(0);
        outerTrackChi.push_back(0);
        outerTrackHits.push_back(0);
        outerTrackRHits.push_back(0);
      }
      // Tight Muon cuts
      if (RecMuon.globalTrack().isNonnull()) {
        globalTrack.push_back(true);
        chiGlobal.push_back(RecMuon.globalTrack()->normalizedChi2());
        globalMuonHits.push_back(RecMuon.globalTrack()->hitPattern().numberOfValidMuonHits());
        matchedStat.push_back(RecMuon.numberOfMatchedStations());
        globalTrckPt.push_back(RecMuon.globalTrack()->pt());
        globalTrckEta.push_back(RecMuon.globalTrack()->eta());
        globalTrckPhi.push_back(RecMuon.globalTrack()->phi());
        tight_TransImpara.push_back(fabs(RecMuon.muonBestTrack()->dxy(pvx)));
        tight_LongPara.push_back(fabs(RecMuon.muonBestTrack()->dz(pvx)));
      } else {
        globalTrack.push_back(false);
        chiGlobal.push_back(0);
        globalMuonHits.push_back(0);
        matchedStat.push_back(0);
        globalTrckPt.push_back(0);
        globalTrckEta.push_back(0);
        globalTrckPhi.push_back(0);
        tight_TransImpara.push_back(0);
        tight_LongPara.push_back(0);
      }

      isolationR04.push_back(
          ((RecMuon.pfIsolationR04().sumChargedHadronPt +
            std::max(0.,
                     RecMuon.pfIsolationR04().sumNeutralHadronEt + RecMuon.pfIsolationR04().sumPhotonEt -
                         (0.5 * RecMuon.pfIsolationR04().sumPUPt))) /
           RecMuon.pt()));

      isolationR03.push_back(
          ((RecMuon.pfIsolationR03().sumChargedHadronPt +
            std::max(0.,
                     RecMuon.pfIsolationR03().sumNeutralHadronEt + RecMuon.pfIsolationR03().sumPhotonEt -
                         (0.5 * RecMuon.pfIsolationR03().sumPUPt))) /
           RecMuon.pt()));

      ecalEnergy.push_back(RecMuon.calEnergy().emS9);
      hcalEnergy.push_back(RecMuon.calEnergy().hadS9);
      hoEnergy.push_back(RecMuon.calEnergy().hoS9);

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
      if (RecMuon.innerTrack().isNonnull()) {
        const reco::Track* pTrack = (RecMuon.innerTrack()).get();
        spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo_, bField, (((verbosity_ / 100) % 10 > 0)));
        if ((RecMuon.p() > pMinMuon_) && (trackID.okHCAL))
          accept = true;

        ecalDetId.push_back((trackID.detIdECAL)());
        hcalDetId.push_back((trackID.detIdHCAL)());
        ehcalDetId.push_back((trackID.detIdEHCAL)());

        HcalDetId check;
        std::pair<bool, HcalDetId> info = spr::propagateHCALBack(pTrack, geo_, bField, (((verbosity_ / 100) % 10 > 0)));
        if (info.first) {
          check = info.second;
        }

        bool okE = trackID.okECAL;
        if (okE) {
          const DetId isoCell(trackID.detIdECAL);
          std::pair<double, bool> e3x3 = spr::eECALmatrix(isoCell,
                                                          barrelRecHitsHandle,
                                                          endcapRecHitsHandle,
                                                          *theEcalChStatus,
                                                          geo_,
                                                          caloTopology,
                                                          sevlv,
                                                          1,
                                                          1,
                                                          -100.0,
                                                          -100.0,
                                                          -500.0,
                                                          500.0,
                                                          false);
          eEcal = e3x3.first;
          okE = e3x3.second;
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HBHEMuon") << "Propagate Track to ECAL: " << okE << ":" << trackID.okECAL << " E " << eEcal;
#endif

        if (trackID.okHCAL) {
          DetId closestCell(trackID.detIdHCAL);
          HcalDetId hcidt(closestCell.rawId());
          if ((hcidt.ieta() == check.ieta()) && (hcidt.iphi() == check.iphi()))
            tmpmatch = true;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HBHEMuon") << "Front " << hcidt << " Back " << info.first << ":" << check << " Match "
                                       << tmpmatch;
#endif

          HcalSubdetector subdet = hcidt.subdet();
          ieta = hcidt.ieta();
          iphi = hcidt.iphi();
          bool hborhe = (std::abs(ieta) == 16);

          eHcal = spr::eHCALmatrix(theHBHETopology_,
                                   closestCell,
                                   hbhe,
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
                              hbhe,
                              ehdepth,
                              maxDepth_,
                              -100.0,
                              -100.0,
                              -100.0,
                              -100.0,
                              -500.0,
                              500.0,
                              useRaw_,
                              depth16HE(ieta, iphi),
                              (((verbosity_ / 1000) % 10) > 0));
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
                tmpC = goodCell(hcid0, pTrack, geo_, bField);
                double enec(ene);
                if (unCorrect_) {
                  double corr = (ignoreHECorr_ && (subdet0 == HcalEndcap)) ? 1.0 : respCorr(DetId(hcid0));
                  if (corr != 0)
                    ene /= corr;
#ifdef EDM_ML_DEBUG
                  HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc_->mergedDepthDetId(hcid0) : hcid0;
                  edm::LogVerbatim("HBHEMuon") << hcid0 << ":" << id << " Corr " << corr;
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
                if ((verbosity_ % 10) > 0)
                  edm::LogVerbatim("HBHEMuon")
                      << hcid0 << " E " << ene << ":" << enec << " L " << actL << " Match " << tmpC;
#endif
              }
            }
          }
#ifdef EDM_ML_DEBUG
          if ((verbosity_ % 10) > 0) {
            edm::LogVerbatim("HBHEMuon") << hcidt << " Match " << tmpmatch << " Depths " << ehdepth.size();
            for (unsigned int k = 0; k < ehdepth.size(); ++k)
              edm::LogVerbatim("HBHEMuon") << " [" << k << ":" << ehdepth[k].second << "] " << matchDepth[k];
          }
#endif
          HcalDetId hotCell;
          spr::eHCALmatrix(geo_, theHBHETopology_, closestCell, hbhe, 1, 1, hotCell, false, useRaw_, false);
          isHot = matchId(closestCell, hotCell);
          if (hotCell != HcalDetId()) {
            subdet = HcalDetId(hotCell).subdet();
            ieta = HcalDetId(hotCell).ieta();
            iphi = HcalDetId(hotCell).iphi();
            hborhe = (std::abs(ieta) == 16);
            std::vector<std::pair<double, int>> ehdepth;
            spr::energyHCALCell(hotCell,
                                hbhe,
                                ehdepth,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                useRaw_,
                                depth16HE(ieta, iphi),
                                false);  //(((verbosity_/1000)%10)>0    ));
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
                  tmpC = goodCell(hcid0, pTrack, geo_, bField);
                  double chg(ene), enec(ene);
                  if (unCorrect_) {
                    double corr = (ignoreHECorr_ && (subdet0 == HcalEndcap)) ? 1.0 : respCorr(DetId(hcid0));
                    if (corr != 0)
                      ene /= corr;
#ifdef EDM_ML_DEBUG
                    HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc_->mergedDepthDetId(hcid0) : hcid0;
                    edm::LogVerbatim("HBHEMuon")
                        << hcid0 << ":" << id << " Corr " << corr << " E " << ene << ":" << enec;
#endif
                  }
                  if (getCharge_) {
                    double gain = gainFactor(conditions, hcid0);
                    if (gain != 0)
                      chg /= gain;
#ifdef EDM_ML_DEBUG
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
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " depth " << depth << " E " << ene << ":" << enec << " C "
                                                 << chg << " L " << actL << " Match " << tmpC;
#endif
                }
              }
            }

            HcalDetId oppCell(subdet, -ieta, iphi, HcalDetId(hotCell).depth());
            std::vector<std::pair<double, int>> ehdeptho;
            spr::energyHCALCell(oppCell,
                                hbhe,
                                ehdeptho,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                useRaw_,
                                depth16HE(-ieta, iphi),
                                false);  //(((verbosity_/1000)%10)>0));
            for (unsigned int i = 0; i < ehdeptho.size(); ++i) {
              HcalSubdetector subdet0 =
                  (hborhe) ? ((ehdeptho[i].second >= depth16HE(-ieta, iphi)) ? HcalEndcap : HcalBarrel) : subdet;
              HcalDetId hcid0(subdet0, -ieta, iphi, ehdeptho[i].second);
              double ene = ehdeptho[i].first;
              if (ene > 0.0) {
                if (!(theHBHETopology_->validHcal(hcid0))) {
                  edm::LogWarning("HBHEMuon") << "(3) Invalid ID " << hcid0 << " with E = " << ene;
                  edm::LogWarning("HBHEMuon") << oppCell << " with " << ehdeptho.size() << " depths:";
                  for (const auto& ehd : ehdeptho)
                    edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
                } else {
                  double chg(ene);
                  if (unCorrect_) {
                    double corr = (ignoreHECorr_ && (subdet0 == HcalEndcap)) ? 1.0 : respCorr(DetId(hcid0));
                    if (corr != 0)
                      ene /= corr;
#ifdef EDM_ML_DEBUG
                    HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc_->mergedDepthDetId(hcid0) : hcid0;
                    edm::LogVerbatim("HBHEMuon")
                        << hcid0 << ":" << id << " Corr " << corr << " E " << ene << ":" << ehdeptho[i].first;
#endif
                  }
                  if (getCharge_) {
                    double gain = gainFactor(conditions, hcid0);
                    if (gain != 0)
                      chg /= gain;
#ifdef EDM_ML_DEBUG
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Gain " << gain << " C " << chg;
#endif
                  }
                  int depth = ehdeptho[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc_->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  cHcalDepthHotBG[depth] += chg;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Depth " << depth << " E " << ene << " C " << chg;
#endif
                }
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HBHEMuon") << "Propagate Track to HCAL: " << trackID.okHCAL << " Match " << tmpmatch
                                     << " Hot " << isHot << " Energy " << eHcal << std::endl;
#endif

      } else {
        ecalDetId.push_back(0);
        hcalDetId.push_back(0);
        ehcalDetId.push_back(0);
      }

      matchedId.push_back(tmpmatch);
      ecal3x3Energy.push_back(eEcal);
      hcal1x1Energy.push_back(eHcal);
      hcal_ieta.push_back(ieta);
      hcal_iphi.push_back(iphi);
      for (int i = 0; i < depthMax_; ++i) {
        hcalDepthEnergy[i].push_back(eHcalDepth[i]);
        hcalDepthActiveLength[i].push_back(activeL[i]);
        hcalDepthEnergyHot[i].push_back(eHcalDepthHot[i]);
        hcalDepthActiveLengthHot[i].push_back(activeHotL[i]);
        hcalDepthEnergyCorr[i].push_back(eHcalDepthC[i]);
        hcalDepthEnergyHotCorr[i].push_back(eHcalDepthHotC[i]);
        hcalDepthChargeHot[i].push_back(cHcalDepthHot[i]);
        hcalDepthChargeHotBG[i].push_back(cHcalDepthHotBG[i]);
        hcalDepthMatch[i].push_back(matchDepth[i]);
        hcalDepthMatchHot[i].push_back(matchDepthHot[i]);
      }
      hcalActiveLength.push_back(activeLengthTot);
      hcalHot.push_back(isHot);
      hcalActiveLengthHot.push_back(activeLengthHotTot);
    }
  }
  if (accept) {
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < hcal_ieta.size(); ++i)
      edm::LogVerbatim("HBHEMuon") << "[" << i << "] ieta/iphi for entry to "
                                   << "HCAL has value of " << hcal_ieta[i] << ":" << hcal_iphi[i];
#endif
    for (unsigned int k = 0; k < muon_is_good.size(); ++k) {
      muon_is_good_ = muon_is_good[k];
      muon_global_ = muon_global[k];
      muon_tracker_ = muon_tracker[k];
      muon_is_tight_ = muon_is_tight[k];
      muon_is_medium_ = muon_is_medium[k];
      ptGlob_ = ptGlob[k];
      etaGlob_ = etaGlob[k];
      phiGlob_ = phiGlob[k];
      energyMuon_ = energyMuon[k];
      pMuon_ = pMuon[k];
      muon_trkKink_ = muon_trkKink[k];
      muon_chi2LocalPosition_ = muon_chi2LocalPosition[k];
      muon_segComp_ = muon_segComp[k];
      trackerLayer_ = trackerLayer[k];
      numPixelLayers_ = numPixelLayers[k];
      tight_PixelHits_ = tight_PixelHits[k];
      innerTrack_ = innerTrack[k];
      outerTrack_ = outerTrack[k];
      globalTrack_ = globalTrack[k];
      chiTracker_ = chiTracker[k];
      dxyTracker_ = dxyTracker[k];
      dzTracker_ = dzTracker[k];
      innerTrackpt_ = innerTrackpt[k];
      innerTracketa_ = innerTracketa[k];
      innerTrackphi_ = innerTrackphi[k];
      tight_validFraction_ = tight_validFraction[k];
      outerTrackChi_ = outerTrackChi[k];
      outerTrackPt_ = outerTrackPt[k];
      outerTrackEta_ = outerTrackEta[k];
      outerTrackPhi_ = outerTrackPhi[k];
      outerTrackHits_ = outerTrackHits[k];
      outerTrackRHits_ = outerTrackRHits[k];
      globalTrckPt_ = globalTrckPt[k];
      globalTrckEta_ = globalTrckEta[k];
      globalTrckPhi_ = globalTrckPhi[k];
      globalMuonHits_ = globalMuonHits[k];
      matchedStat_ = matchedStat[k];
      chiGlobal_ = chiGlobal[k];
      tight_LongPara_ = tight_LongPara[k];
      tight_TransImpara_ = tight_TransImpara[k];
      isolationR04_ = isolationR04[k];
      isolationR03_ = isolationR03[k];
      ecalEnergy_ = ecalEnergy[k];
      hcalEnergy_ = hcalEnergy[k];
      hoEnergy_ = hoEnergy[k];
      matchedId_ = matchedId[k];
      hcalHot_ = hcalHot[k];
      ecal3x3Energy_ = ecal3x3Energy[k];
      hcal1x1Energy_ = hcal1x1Energy[k];
      ecalDetId_ = ecalDetId[k];
      hcalDetId_ = hcalDetId[k];
      ehcalDetId_ = ehcalDetId[k];
      hcal_ieta_ = hcal_ieta[k];
      hcal_iphi_ = hcal_iphi[k];
      for (int i = 0; i < depthMax_; ++i) {
        hcalDepthEnergy_[i] = hcalDepthEnergy[i][k];
        hcalDepthActiveLength_[i] = hcalDepthActiveLength[i][k];
        hcalDepthEnergyHot_[i] = hcalDepthEnergyHot[i][k];
        hcalDepthActiveLengthHot_[i] = hcalDepthActiveLengthHot[i][k];
        hcalDepthChargeHot_[i] = hcalDepthChargeHot[i][k];
        hcalDepthChargeHotBG_[i] = hcalDepthChargeHotBG[i][k];
        hcalDepthEnergyCorr_[i] = hcalDepthEnergyCorr[i][k];
        hcalDepthEnergyHotCorr_[i] = hcalDepthEnergyHotCorr[i][k];
        hcalDepthMatch_[i] = hcalDepthMatch[i][k];
        hcalDepthMatchHot_[i] = hcalDepthMatchHot[i][k];
      }
      hcalActiveLength_ = hcalActiveLength[k];
      hcalActiveLengthHot_ = hcalActiveLengthHot[k];
      tree_->Fill();
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HcalHBHEMuonAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("TREE", "TREE");
  tree_->Branch("Event_No", &eventNumber_);
  tree_->Branch("Run_No", &runNumber_);
  tree_->Branch("LumiNumber", &lumiNumber_);
  tree_->Branch("BXNumber", &bxNumber_);
  tree_->Branch("GoodVertex", &goodVertex_);
  tree_->Branch("PF_Muon", &muon_is_good_);
  tree_->Branch("Global_Muon", &muon_global_);
  tree_->Branch("Tracker_muon", &muon_tracker_);
  tree_->Branch("MuonIsTight", &muon_is_tight_);
  tree_->Branch("MuonIsMedium", &muon_is_medium_);
  tree_->Branch("pt_of_muon", &ptGlob_);
  tree_->Branch("eta_of_muon", &etaGlob_);
  tree_->Branch("phi_of_muon", &phiGlob_);
  tree_->Branch("energy_of_muon", &energyMuon_);
  tree_->Branch("p_of_muon", &pMuon_);
  tree_->Branch("muon_trkKink", &muon_trkKink_);
  tree_->Branch("muon_chi2LocalPosition", &muon_chi2LocalPosition_);
  tree_->Branch("muon_segComp", &muon_segComp_);

  tree_->Branch("TrackerLayer", &trackerLayer_);
  tree_->Branch("NumPixelLayers", &numPixelLayers_);
  tree_->Branch("InnerTrackPixelHits", &tight_PixelHits_);
  tree_->Branch("innerTrack", &innerTrack_);
  tree_->Branch("chiTracker", &chiTracker_);
  tree_->Branch("DxyTracker", &dxyTracker_);
  tree_->Branch("DzTracker", &dzTracker_);
  tree_->Branch("innerTrackpt", &innerTrackpt_);
  tree_->Branch("innerTracketa", &innerTracketa_);
  tree_->Branch("innerTrackphi", &innerTrackphi_);
  tree_->Branch("tight_validFraction", &tight_validFraction_);

  tree_->Branch("OuterTrack", &outerTrack_);
  tree_->Branch("OuterTrackChi", &outerTrackChi_);
  tree_->Branch("OuterTrackPt", &outerTrackPt_);
  tree_->Branch("OuterTrackEta", &outerTrackEta_);
  tree_->Branch("OuterTrackPhi", &outerTrackPhi_);
  tree_->Branch("OuterTrackHits", &outerTrackHits_);
  tree_->Branch("OuterTrackRHits", &outerTrackRHits_);

  tree_->Branch("GlobalTrack", &globalTrack_);
  tree_->Branch("GlobalTrckPt", &globalTrckPt_);
  tree_->Branch("GlobalTrckEta", &globalTrckEta_);
  tree_->Branch("GlobalTrckPhi", &globalTrckPhi_);
  tree_->Branch("Global_Muon_Hits", &globalMuonHits_);
  tree_->Branch("MatchedStations", &matchedStat_);
  tree_->Branch("GlobTrack_Chi", &chiGlobal_);
  tree_->Branch("Tight_LongitudinalImpactparameter", &tight_LongPara_);
  tree_->Branch("Tight_TransImpactparameter", &tight_TransImpara_);

  tree_->Branch("IsolationR04", &isolationR04_);
  tree_->Branch("IsolationR03", &isolationR03_);
  tree_->Branch("ecal_3into3", &ecalEnergy_);
  tree_->Branch("hcal_3into3", &hcalEnergy_);
  tree_->Branch("tracker_3into3", &hoEnergy_);

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

  tree_->Branch("hltresults", &hltresults_);
  tree_->Branch("all_triggers", &all_triggers_);
}

// ------------ method called when starting to processes a run  ------------
void HcalHBHEMuonAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  hdc_ = &iSetup.getData(tok_ddrec_);
  actHB.clear();
  actHE.clear();
  actHB = hdc_->getThickActive(0);
  actHE = hdc_->getThickActive(1);
#ifdef EDM_ML_DEBUG
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
#endif

  bool changed = true;
  all_triggers_.clear();
  if (hltConfig_.init(iRun, iSetup, "HLT", changed)) {
    // if init returns TRUE, initialisation has succeeded!
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "HLT config with process name HLT successfully extracted";
#endif
    unsigned int ntriggers = hltConfig_.size();
    for (unsigned int t = 0; t < ntriggers; ++t) {
      std::string hltname(hltConfig_.triggerName(t));
      for (unsigned int ik = 0; ik < triggers_.size(); ++ik) {
        if (hltname.find(triggers_[ik]) != std::string::npos) {
          all_triggers_.push_back(hltname);
          break;
        }
      }
    }  //loop over ntriggers
    edm::LogVerbatim("HBHEMuon") << "All triggers size in begin run " << all_triggers_.size();
  } else {
    edm::LogError("HBHEMuon") << "Error! HLT config extraction with process name HLT failed";
  }

  theHBHETopology_ = &iSetup.getData(tok_htopo_);
  const HcalRespCorrs* resp = &iSetup.getData(tok_respcorr_);
  respCorrs_ = new HcalRespCorrs(*resp);
  respCorrs_->setTopo(theHBHETopology_);
  geo_ = &iSetup.getData(tok_geom_);

  // Write correction factors for all HB/HE events
  if (writeRespCorr_) {
    const HcalGeometry* gHcal = static_cast<const HcalGeometry*>(geo_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
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

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalHBHEMuonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hlTriggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("labelEBRecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("labelEERecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("labelHBHERecHit", edm::InputTag("hbhereco"));
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelMuon", "muons");
  std::vector<std::string> trig = {};
  desc.add<std::vector<std::string>>("triggers", trig);
  desc.add<double>("pMinMuon", 5.0);
  desc.addUntracked<int>("verbosity", 0);
  desc.add<int>("useRaw", 0);
  desc.add<bool>("unCorrect", true);
  desc.add<bool>("getCharge", true);
  desc.add<bool>("collapseDepth", false);
  desc.add<bool>("isItPlan1", false);
  desc.addUntracked<bool>("ignoreHECorr", false);
  desc.addUntracked<bool>("isItPreRecHit", false);
  desc.addUntracked<std::string>("moduleName", "");
  desc.addUntracked<std::string>("processName", "");
  desc.addUntracked<int>("maxDepth", 7);
  desc.addUntracked<std::string>("fileInCorr", "");
  desc.addUntracked<bool>("writeRespCorr", false);
  descriptions.add("hcalHBHEMuon", desc);
}

void HcalHBHEMuonAnalyzer::clearVectors() {
  ///clearing vectots
  eventNumber_ = -99999;
  runNumber_ = -99999;
  lumiNumber_ = -99999;
  bxNumber_ = -99999;
  goodVertex_ = -99999;

  muon_is_good_ = false;
  muon_global_ = false;
  muon_tracker_ = false;
  ptGlob_ = 0;
  etaGlob_ = 0;
  phiGlob_ = 0;
  energyMuon_ = 0;
  pMuon_ = 0;
  muon_trkKink_ = 0;
  muon_chi2LocalPosition_ = 0;
  muon_segComp_ = 0;
  muon_is_tight_ = false;
  muon_is_medium_ = false;

  trackerLayer_ = 0;
  numPixelLayers_ = 0;
  tight_PixelHits_ = 0;
  innerTrack_ = false;
  chiTracker_ = 0;
  dxyTracker_ = 0;
  dzTracker_ = 0;
  innerTrackpt_ = 0;
  innerTracketa_ = 0;
  innerTrackphi_ = 0;
  tight_validFraction_ = 0;

  outerTrack_ = false;
  outerTrackPt_ = 0;
  outerTrackEta_ = 0;
  outerTrackPhi_ = 0;
  outerTrackHits_ = 0;
  outerTrackRHits_ = 0;
  outerTrackChi_ = 0;

  globalTrack_ = false;
  globalTrckPt_ = 0;
  globalTrckEta_ = 0;
  globalTrckPhi_ = 0;
  globalMuonHits_ = 0;
  matchedStat_ = 0;
  chiGlobal_ = 0;
  tight_LongPara_ = 0;
  tight_TransImpara_ = 0;

  isolationR04_ = 0;
  isolationR03_ = 0;
  ecalEnergy_ = 0;
  hcalEnergy_ = 0;
  hoEnergy_ = 0;
  matchedId_ = false;
  hcalHot_ = false;
  ecal3x3Energy_ = 0;
  hcal1x1Energy_ = 0;
  ecalDetId_ = 0;
  hcalDetId_ = 0;
  ehcalDetId_ = 0;
  hcal_ieta_ = 0;
  hcal_iphi_ = 0;
  for (int i = 0; i < maxDepth_; ++i) {
    hcalDepthEnergy_[i] = 0;
    hcalDepthActiveLength_[i] = 0;
    hcalDepthEnergyHot_[i] = 0;
    hcalDepthActiveLengthHot_[i] = 0;
    hcalDepthChargeHot_[i] = 0;
    hcalDepthChargeHotBG_[i] = 0;
    hcalDepthEnergyCorr_[i] = 0;
    hcalDepthEnergyHotCorr_[i] = 0;
    hcalDepthMatch_[i] = false;
    hcalDepthMatchHot_[i] = false;
  }
  hcalActiveLength_ = 0;
  hcalActiveLengthHot_ = 0;
  hltresults_.clear();
}

int HcalHBHEMuonAnalyzer::matchId(const HcalDetId& id1, const HcalDetId& id2) {
  HcalDetId kd1(id1.subdet(), id1.ieta(), id1.iphi(), 1);
  HcalDetId kd2(id1.subdet(), id2.ieta(), id2.iphi(), 1);
  int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalHBHEMuonAnalyzer::activeLength(const DetId& id_) {
  HcalDetId id(id_);
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

bool HcalHBHEMuonAnalyzer::isGoodVertex(const reco::Vertex& vtx) {
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

double HcalHBHEMuonAnalyzer::respCorr(const DetId& id) {
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

double HcalHBHEMuonAnalyzer::gainFactor(const HcalDbService* conditions, const HcalDetId& id) {
  double gain(0.0);
  const HcalCalibrations& calibs = conditions->getHcalCalibrations(id);
  for (int capid = 0; capid < 4; ++capid)
    gain += (0.25 * calibs.respcorrgain(capid));
  return gain;
}

int HcalHBHEMuonAnalyzer::depth16HE(int ieta, int iphi) {
  // Transition between HB/HE is special
  // For Run 1 or for Plan1 standard reconstruction it is 3
  // For runs beyond 2018 or in Plan1 for HEP17 it is 4
  int zside = (ieta > 0) ? 1 : -1;
  int depth = theHBHETopology_->dddConstants()->getMinDepth(1, 16, iphi, zside);
  if (isItPlan1_ && (!isItPreRecHit_))
    depth = 3;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Plan1 " << isItPlan1_ << " PreRecHit " << isItPreRecHit_ << " phi " << iphi
                               << " depth " << depth;
#endif
  return depth;
}

bool HcalHBHEMuonAnalyzer::goodCell(const HcalDetId& hcid,
                                    const reco::Track* pTrack,
                                    const CaloGeometry* geo,
                                    const MagneticField* bField) {
  std::pair<double, double> rz = hdc_->getRZ(hcid);
  bool typeRZ = (hcid.subdet() == HcalEndcap) ? false : true;
  bool match = spr::propagateHCAL(pTrack, geo, bField, typeRZ, rz, (((verbosity_ / 10000) % 10) > 0));
  return match;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalHBHEMuonAnalyzer);
