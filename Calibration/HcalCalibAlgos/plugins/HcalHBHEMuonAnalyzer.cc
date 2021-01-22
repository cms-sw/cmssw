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
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
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
  edm::Service<TFileService> fs;
  const edm::InputTag hlTriggerResults_;
  const edm::InputTag labelEBRecHit_, labelEERecHit_, labelHBHERecHit_;
  const std::string labelVtx_, labelMuon_, fileInCorr_;
  const std::vector<std::string> triggers_;
  const int verbosity_, useRaw_;
  const bool unCorrect_, collapseDepth_, isItPlan1_;
  const bool ignoreHECorr_, isItPreRecHit_;
  const bool getCharge_, writeRespCorr_;
  bool mergedDepth_, useMyCorr_;
  int maxDepth_, kount_;

  const HcalDDDRecConstants* hdc_;
  const HcalTopology* theHBHETopology_;
  const CaloGeometry* geo_;
  HcalRespCorrs* respCorrs_;

  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<reco::VertexCollection> tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respcorr_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_chan_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_topo_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbservice_;

  //////////////////////////////////////////////////////
  static const int depthMax_ = 7;
  TTree* tree_;
  unsigned int runNumber_, eventNumber_, lumiNumber_, bxNumber_;
  unsigned int goodVertex_;
  std::vector<bool> muon_is_good_, muon_global_, muon_tracker_;
  std::vector<bool> muon_is_tight_, muon_is_medium_;
  std::vector<double> ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  std::vector<float> muon_trkKink, muon_chi2LocalPosition, muon_segComp;
  std::vector<int> trackerLayer_, numPixelLayers_, tight_PixelHits_;
  std::vector<bool> innerTrack_, outerTrack_, globalTrack_;
  std::vector<double> chiTracker_, dxyTracker_, dzTracker_;
  std::vector<double> innerTrackpt_, innerTracketa_, innerTrackphi_;
  std::vector<double> tight_validFraction_, outerTrackChi_;
  std::vector<double> outerTrackPt_, outerTrackEta_, outerTrackPhi_;
  std::vector<int> outerTrackHits_, outerTrackRHits_;
  std::vector<double> globalTrckPt_, globalTrckEta_, globalTrckPhi_;
  std::vector<int> globalMuonHits_, matchedStat_;
  std::vector<double> chiGlobal_, tight_LongPara_, tight_TransImpara_;
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
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      useRaw_(iConfig.getParameter<int>("useRaw")),
      unCorrect_(iConfig.getParameter<bool>("unCorrect")),
      collapseDepth_(iConfig.getParameter<bool>("collapseDepth")),
      isItPlan1_(iConfig.getParameter<bool>("isItPlan1")),
      ignoreHECorr_(iConfig.getUntrackedParameter<bool>("ignoreHECorr", false)),
      isItPreRecHit_(iConfig.getUntrackedParameter<bool>("isItPreRecHit", false)),
      getCharge_(iConfig.getParameter<bool>("getCharge")),
      writeRespCorr_(iConfig.getUntrackedParameter<bool>("writeRespCorr", false)),
      hdc_(nullptr),
      theHBHETopology_(nullptr),
      respCorrs_(nullptr) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  kount_ = 0;
  maxDepth_ = iConfig.getUntrackedParameter<int>("maxDepth", 4);
  if (maxDepth_ > depthMax_)
    maxDepth_ = depthMax_;
  else if (maxDepth_ < 1)
    maxDepth_ = 4;
  std::string modnam = iConfig.getUntrackedParameter<std::string>("moduleName", "");
  std::string procnm = iConfig.getUntrackedParameter<std::string>("processName", "");

  mergedDepth_ = (!isItPreRecHit_) || (collapseDepth_);
  tok_trigRes_ = consumes<edm::TriggerResults>(hlTriggerResults_);
  tok_EB_ = consumes<EcalRecHitCollection>(labelEBRecHit_);
  tok_EE_ = consumes<EcalRecHitCollection>(labelEERecHit_);
  tok_HBHE_ = consumes<HBHERecHitCollection>(labelHBHERecHit_);
  if (modnam.empty()) {
    tok_Vtx_ = consumes<reco::VertexCollection>(labelVtx_);
    tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);
    edm::LogVerbatim("HBHEMuon") << "Labels used: Trig " << hlTriggerResults_ << " Vtx " << labelVtx_ << " EB "
                                 << labelEBRecHit_ << " EE " << labelEERecHit_ << " HBHE " << labelHBHERecHit_ << " MU "
                                 << labelMuon_;
  } else {
    tok_Vtx_ = consumes<reco::VertexCollection>(edm::InputTag(modnam, labelVtx_, procnm));
    tok_Muon_ = consumes<reco::MuonCollection>(edm::InputTag(modnam, labelMuon_, procnm));
    edm::LogVerbatim("HBHEMuon") << "Labels used Trig " << hlTriggerResults_ << "\n  Vtx  "
                                 << edm::InputTag(modnam, labelVtx_, procnm) << "\n  EB   " << labelEBRecHit_
                                 << "\n  EE   " << labelEERecHit_ << "\n  HBHE " << labelHBHERecHit_ << "\n  MU   "
                                 << edm::InputTag(modnam, labelMuon_, procnm);
  }

  tok_ddrec_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  tok_respcorr_ = esConsumes<HcalRespCorrs, HcalRespCorrsRcd, edm::Transition::BeginRun>();
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_chan_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  tok_sevlv_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();
  tok_topo_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_dbservice_ = esConsumes<HcalDbService, HcalDbRecord>();

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

// ------------ method called for each event  ------------
void HcalHBHEMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++kount_;
  clearVectors();
  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_ = iEvent.id().luminosityBlock();
  bxNumber_ = iEvent.bunchCrossing();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber_ << " Event " << eventNumber_ << " Lumi " << lumiNumber_ << " BX "
                               << bxNumber_ << std::endl;
#endif
  edm::Handle<edm::TriggerResults> _Triggers;
  iEvent.getByToken(tok_trigRes_, _Triggers);
#ifdef EDM_ML_DEBUG
  if ((verbosity_ / 10000) % 10 > 0)
    edm::LogVerbatim("HBHEMuon") << "Size of all triggers " << all_triggers_.size() << std::endl;
#endif
  int Ntriggers = all_triggers_.size();
#ifdef EDM_ML_DEBUG
  if ((verbosity_ / 10000) % 10 > 0)
    edm::LogVerbatim("HBHEMuon") << "Size of HLT MENU: " << _Triggers->size() << std::endl;
#endif
  if (_Triggers.isValid()) {
    const edm::TriggerNames& triggerNames_ = iEvent.triggerNames(*_Triggers);
    std::vector<int> index;
    for (int i = 0; i < Ntriggers; i++) {
      index.push_back(triggerNames_.triggerIndex(all_triggers_[i]));
      int triggerSize = int(_Triggers->size());
#ifdef EDM_ML_DEBUG
      if ((verbosity_ / 10000) % 10 > 0)
        edm::LogVerbatim("HBHEMuon") << "outside loop " << index[i] << "\ntriggerSize " << triggerSize << std::endl;
#endif
      if (index[i] < triggerSize) {
        hltresults_.push_back(_Triggers->accept(index[i]));
#ifdef EDM_ML_DEBUG
        if ((verbosity_ / 10000) % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Trigger_info " << triggerSize << " triggerSize " << index[i]
                                       << " trigger_index " << hltresults_.at(i) << " hltresult" << std::endl;
#endif
      } else {
        if ((verbosity_ / 10000) % 10 > 0)
          edm::LogVerbatim("HBHEMuon") << "Requested HLT path \""
                                       << "\" does not exist\n";
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
  edm::Handle<reco::VertexCollection> vtx;
  iEvent.getByToken(tok_Vtx_, vtx);

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_HBHE_, hbhe);

  edm::Handle<reco::MuonCollection> _Muon;
  iEvent.getByToken(tok_Muon_, _Muon);

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
  if (_Muon.isValid() && barrelRecHitsHandle.isValid() && endcapRecHitsHandle.isValid() && hbhe.isValid()) {
    for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon != _Muon->end(); ++RecMuon) {
      muon_is_good_.push_back(RecMuon->isPFMuon());
      muon_global_.push_back(RecMuon->isGlobalMuon());
      muon_tracker_.push_back(RecMuon->isTrackerMuon());
      ptGlob_.push_back((RecMuon)->pt());
      etaGlob_.push_back(RecMuon->eta());
      phiGlob_.push_back(RecMuon->phi());
      energyMuon_.push_back(RecMuon->energy());
      pMuon_.push_back(RecMuon->p());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HBHEMuon") << "Energy:" << RecMuon->energy() << " P:" << RecMuon->p() << std::endl;
#endif
      muon_is_tight_.push_back(muon::isTightMuon(*RecMuon, *firstGoodVertex));
      muon_is_medium_.push_back(muon::isMediumMuon(*RecMuon));
      muon_trkKink.push_back(RecMuon->combinedQuality().trkKink);
      muon_chi2LocalPosition.push_back(RecMuon->combinedQuality().chi2LocalPosition);
      muon_segComp.push_back(muon::segmentCompatibility(*RecMuon));
      // acessing tracker hits info
      if (RecMuon->track().isNonnull()) {
        trackerLayer_.push_back(RecMuon->track()->hitPattern().trackerLayersWithMeasurement());
      } else {
        trackerLayer_.push_back(-1);
      }
      if (RecMuon->innerTrack().isNonnull()) {
        innerTrack_.push_back(true);
        numPixelLayers_.push_back(RecMuon->innerTrack()->hitPattern().pixelLayersWithMeasurement());
        chiTracker_.push_back(RecMuon->innerTrack()->normalizedChi2());
        dxyTracker_.push_back(fabs(RecMuon->innerTrack()->dxy(pvx)));
        dzTracker_.push_back(fabs(RecMuon->innerTrack()->dz(pvx)));
        innerTrackpt_.push_back(RecMuon->innerTrack()->pt());
        innerTracketa_.push_back(RecMuon->innerTrack()->eta());
        innerTrackphi_.push_back(RecMuon->innerTrack()->phi());
        tight_PixelHits_.push_back(RecMuon->innerTrack()->hitPattern().numberOfValidPixelHits());
        tight_validFraction_.push_back(RecMuon->innerTrack()->validFraction());
      } else {
        innerTrack_.push_back(false);
        numPixelLayers_.push_back(0);
        chiTracker_.push_back(0);
        dxyTracker_.push_back(0);
        dzTracker_.push_back(0);
        innerTrackpt_.push_back(0);
        innerTracketa_.push_back(0);
        innerTrackphi_.push_back(0);
        tight_PixelHits_.push_back(0);
        tight_validFraction_.push_back(-99);
      }
      // outer track info
      if (RecMuon->outerTrack().isNonnull()) {
        outerTrack_.push_back(true);
        outerTrackPt_.push_back(RecMuon->outerTrack()->pt());
        outerTrackEta_.push_back(RecMuon->outerTrack()->eta());
        outerTrackPhi_.push_back(RecMuon->outerTrack()->phi());
        outerTrackChi_.push_back(RecMuon->outerTrack()->normalizedChi2());
        outerTrackHits_.push_back(RecMuon->outerTrack()->numberOfValidHits());
        outerTrackRHits_.push_back(RecMuon->outerTrack()->recHitsSize());
      } else {
        outerTrack_.push_back(false);
        outerTrackPt_.push_back(0);
        outerTrackEta_.push_back(0);
        outerTrackPhi_.push_back(0);
        outerTrackChi_.push_back(0);
        outerTrackHits_.push_back(0);
        outerTrackRHits_.push_back(0);
      }
      // Tight Muon cuts
      if (RecMuon->globalTrack().isNonnull()) {
        globalTrack_.push_back(true);
        chiGlobal_.push_back(RecMuon->globalTrack()->normalizedChi2());
        globalMuonHits_.push_back(RecMuon->globalTrack()->hitPattern().numberOfValidMuonHits());
        matchedStat_.push_back(RecMuon->numberOfMatchedStations());
        globalTrckPt_.push_back(RecMuon->globalTrack()->pt());
        globalTrckEta_.push_back(RecMuon->globalTrack()->eta());
        globalTrckPhi_.push_back(RecMuon->globalTrack()->phi());
        tight_TransImpara_.push_back(fabs(RecMuon->muonBestTrack()->dxy(pvx)));
        tight_LongPara_.push_back(fabs(RecMuon->muonBestTrack()->dz(pvx)));
      } else {
        globalTrack_.push_back(false);
        chiGlobal_.push_back(0);
        globalMuonHits_.push_back(0);
        matchedStat_.push_back(0);
        globalTrckPt_.push_back(0);
        globalTrckEta_.push_back(0);
        globalTrckPhi_.push_back(0);
        tight_TransImpara_.push_back(0);
        tight_LongPara_.push_back(0);
      }

      isolationR04_.push_back(
          ((RecMuon->pfIsolationR04().sumChargedHadronPt +
            std::max(0.,
                     RecMuon->pfIsolationR04().sumNeutralHadronEt + RecMuon->pfIsolationR04().sumPhotonEt -
                         (0.5 * RecMuon->pfIsolationR04().sumPUPt))) /
           RecMuon->pt()));

      isolationR03_.push_back(
          ((RecMuon->pfIsolationR03().sumChargedHadronPt +
            std::max(0.,
                     RecMuon->pfIsolationR03().sumNeutralHadronEt + RecMuon->pfIsolationR03().sumPhotonEt -
                         (0.5 * RecMuon->pfIsolationR03().sumPUPt))) /
           RecMuon->pt()));

      ecalEnergy_.push_back(RecMuon->calEnergy().emS9);
      hcalEnergy_.push_back(RecMuon->calEnergy().hadS9);
      hoEnergy_.push_back(RecMuon->calEnergy().hoS9);

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
      if (RecMuon->innerTrack().isNonnull()) {
        const reco::Track* pTrack = (RecMuon->innerTrack()).get();
        spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo_, bField, (((verbosity_ / 100) % 10 > 0)));
        if ((RecMuon->p() > 10.0) && (trackID.okHCAL))
          accept = true;

        ecalDetId_.push_back((trackID.detIdECAL)());
        hcalDetId_.push_back((trackID.detIdHCAL)());
        ehcalDetId_.push_back((trackID.detIdEHCAL)());

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
        ecalDetId_.push_back(0);
        hcalDetId_.push_back(0);
        ehcalDetId_.push_back(0);
      }

      matchedId_.push_back(tmpmatch);
      ecal3x3Energy_.push_back(eEcal);
      hcal1x1Energy_.push_back(eHcal);
      hcal_ieta_.push_back(ieta);
      hcal_iphi_.push_back(iphi);
      for (int i = 0; i < depthMax_; ++i) {
        hcalDepthEnergy_[i].push_back(eHcalDepth[i]);
        hcalDepthActiveLength_[i].push_back(activeL[i]);
        hcalDepthEnergyHot_[i].push_back(eHcalDepthHot[i]);
        hcalDepthActiveLengthHot_[i].push_back(activeHotL[i]);
        hcalDepthEnergyCorr_[i].push_back(eHcalDepthC[i]);
        hcalDepthEnergyHotCorr_[i].push_back(eHcalDepthHotC[i]);
        hcalDepthChargeHot_[i].push_back(cHcalDepthHot[i]);
        hcalDepthChargeHotBG_[i].push_back(cHcalDepthHotBG[i]);
        hcalDepthMatch_[i].push_back(matchDepth[i]);
        hcalDepthMatchHot_[i].push_back(matchDepthHot[i]);
      }
      hcalActiveLength_.push_back(activeLengthTot);
      hcalHot_.push_back(isHot);
      hcalActiveLengthHot_.push_back(activeLengthHotTot);
    }
  }
  if (accept) {
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < hcal_ieta_.size(); ++i)
      edm::LogVerbatim("HBHEMuon") << "[" << i << "] ieta/iphi for entry to "
                                   << "HCAL has value of " << hcal_ieta_[i] << ":" << hcal_iphi_[i];
#endif
    tree_->Fill();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HcalHBHEMuonAnalyzer::beginJob() {
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
  tree_->Branch("muon_trkKink", &muon_trkKink);
  tree_->Branch("muon_chi2LocalPosition", &muon_chi2LocalPosition);
  tree_->Branch("muon_segComp", &muon_segComp);

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
    edm::LogVerbatim("HBHEMuon") << "HLT config with process name "
                                 << "HLT"
                                 << " successfully extracted" << std::endl;
#endif
    unsigned int ntriggers = hltConfig_.size();
    for (unsigned int t = 0; t < ntriggers; ++t) {
      std::string hltname(hltConfig_.triggerName(t));
      for (unsigned int ik = 0; ik < 6; ++ik) {
        if (hltname.find(triggers_[ik]) != std::string::npos) {
          all_triggers_.push_back(hltname);
          break;
        }
      }
    }  //loop over ntriggers
    edm::LogVerbatim("HBHEMuon") << "All triggers size in begin run " << all_triggers_.size() << std::endl;
  } else {
    edm::LogError("HBHEMuon") << "Error! HLT config extraction with process "
                              << "name HLT failed";
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
  std::vector<std::string> trig = {"HLT_IsoMu17", "HLT_IsoMu20", "HLT_IsoMu24", "HLT_IsoMu27", "HLT_Mu45", "HLT_Mu50"};
  desc.add<std::vector<std::string>>("triggers", trig);
  desc.addUntracked<int>("verbosity", 0);
  desc.add<int>("useRaw", 0);
  desc.add<bool>("unCorrect", false);
  desc.add<bool>("getCharge", false);
  desc.add<bool>("collapseDepth", false);
  desc.add<bool>("isItPlan1", false);
  desc.addUntracked<bool>("ignoreHECorr", false);
  desc.addUntracked<bool>("isItPreRecHit", false);
  desc.addUntracked<std::string>("moduleName", "");
  desc.addUntracked<std::string>("processName", "");
  desc.addUntracked<int>("maxDepth", 4);
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

  muon_is_good_.clear();
  muon_global_.clear();
  muon_tracker_.clear();
  ptGlob_.clear();
  etaGlob_.clear();
  phiGlob_.clear();
  energyMuon_.clear();
  pMuon_.clear();
  muon_trkKink.clear();
  muon_chi2LocalPosition.clear();
  muon_segComp.clear();
  muon_is_tight_.clear();
  muon_is_medium_.clear();

  trackerLayer_.clear();
  numPixelLayers_.clear();
  tight_PixelHits_.clear();
  innerTrack_.clear();
  chiTracker_.clear();
  dxyTracker_.clear();
  dzTracker_.clear();
  innerTrackpt_.clear();
  innerTracketa_.clear();
  innerTrackphi_.clear();
  tight_validFraction_.clear();

  outerTrack_.clear();
  outerTrackPt_.clear();
  outerTrackEta_.clear();
  outerTrackPhi_.clear();
  outerTrackHits_.clear();
  outerTrackRHits_.clear();
  outerTrackChi_.clear();

  globalTrack_.clear();
  globalTrckPt_.clear();
  globalTrckEta_.clear();
  globalTrckPhi_.clear();
  globalMuonHits_.clear();
  matchedStat_.clear();
  chiGlobal_.clear();
  tight_LongPara_.clear();
  tight_TransImpara_.clear();

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
  for (int i = 0; i < maxDepth_; ++i) {
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
