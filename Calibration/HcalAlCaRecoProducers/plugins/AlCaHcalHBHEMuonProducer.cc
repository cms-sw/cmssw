#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include "TPRegexp.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/HcalCalibObjects/interface/HcalHBHEMuonVariables.h"
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

namespace alcaHcalHBHEMuon {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace alcaHcalHBHEMuon

class AlCaHcalHBHEMuonProducer : public edm::stream::EDProducer<edm::GlobalCache<alcaHcalHBHEMuon::Counters>> {
public:
  explicit AlCaHcalHBHEMuonProducer(const edm::ParameterSet&, const alcaHcalHBHEMuon::Counters*);
  ~AlCaHcalHBHEMuonProducer() override = default;

  static std::unique_ptr<alcaHcalHBHEMuon::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<alcaHcalHBHEMuon::Counters>();
  }

  void produce(edm::Event&, const edm::EventSetup&) override;

  void endStream() override;

  static void globalEndJob(const alcaHcalHBHEMuon::Counters* counters);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  int matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&, const HcalDDDRecConstants* hdc);
  bool isGoodVertex(const reco::Vertex& vtx);
  double respCorr(const DetId& id, const HcalRespCorrs* respCorrs);
  double gainFactor(const HcalDbService* dbserv, const HcalDetId& id);
  int depth16HE(int ieta, int iphi, const HcalTopology* theHBHETopology);
  bool goodCell(const HcalDetId& hcid,
                const reco::Track* pTrack,
                const CaloGeometry* geo,
                const MagneticField* bField,
                const HcalDDDRecConstants* hdc);

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  const std::vector<std::string> trigNames_;
  const std::string processName_;
  const edm::InputTag triggerResults_;
  const edm::InputTag labelEBRecHit_, labelEERecHit_, labelHBHERecHit_;
  const std::string labelVtx_, labelMuon_, labelHBHEMuon_;
  const bool collapseDepth_, isItPlan1_;
  const int verbosity_;
  const bool isItPreRecHit_, writeRespCorr_;
  const std::string fileInCorr_;
  const int maxDepth_;
  const bool mergedDepth_;

  bool useMyCorr_;
  int nRun_, nAll_, nGood_;

  const edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  const edm::EDGetTokenT<reco::VertexCollection> tok_Vtx_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  const edm::EDGetTokenT<HBHERecHitCollection> tok_HBHE_;
  const edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;

  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec0_;
  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec1_;
  const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respcorr0_;
  const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respcorr1_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom0_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom1_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo0_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo1_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_chan_;
  const edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_topo_;
  const edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbservice_;

  //////////////////////////////////////////////////////
  static const int depthMax_ = 7;
  std::vector<std::string> all_triggers_;
  std::vector<int> hltresults_;

  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB, actHE;
  std::map<DetId, double> corrValue_;
  ////////////////////////////////////////////////////////////
};

AlCaHcalHBHEMuonProducer::AlCaHcalHBHEMuonProducer(const edm::ParameterSet& iConfig, const alcaHcalHBHEMuon::Counters*)
    : trigNames_(iConfig.getParameter<std::vector<std::string>>("triggers")),
      processName_(iConfig.getParameter<std::string>("processName")),
      triggerResults_(iConfig.getParameter<edm::InputTag>("triggerResults")),
      labelEBRecHit_(iConfig.getParameter<edm::InputTag>("labelEBRecHit")),
      labelEERecHit_(iConfig.getParameter<edm::InputTag>("labelEERecHit")),
      labelHBHERecHit_(iConfig.getParameter<edm::InputTag>("labelHBHERecHit")),
      labelVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelMuon_(iConfig.getParameter<std::string>("labelMuon")),
      labelHBHEMuon_(iConfig.getParameter<std::string>("labelHBHEMuon")),
      collapseDepth_(iConfig.getParameter<bool>("collapseDepth")),
      isItPlan1_(iConfig.getParameter<bool>("isItPlan1")),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      isItPreRecHit_(iConfig.getUntrackedParameter<bool>("isItPreRecHit", false)),
      writeRespCorr_(iConfig.getUntrackedParameter<bool>("writeRespCorr", false)),
      fileInCorr_(iConfig.getUntrackedParameter<std::string>("fileInCorr", "")),
      maxDepth_(iConfig.getUntrackedParameter<int>("maxDepth", 4)),
      mergedDepth_((!isItPreRecHit_) || (collapseDepth_)),
      nRun_(0),
      nAll_(0),
      nGood_(0),
      tok_trigRes_(consumes<edm::TriggerResults>(triggerResults_)),
      tok_Vtx_(consumes<reco::VertexCollection>(labelVtx_)),
      tok_EB_(consumes<EcalRecHitCollection>(labelEBRecHit_)),
      tok_EE_(consumes<EcalRecHitCollection>(labelEERecHit_)),
      tok_HBHE_(consumes<HBHERecHitCollection>(labelHBHERecHit_)),
      tok_Muon_(consumes<reco::MuonCollection>(labelMuon_)),
      tok_ddrec0_(esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      tok_ddrec1_(esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>()),
      tok_respcorr0_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd, edm::Transition::BeginRun>()),
      tok_respcorr1_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd>()),
      tok_geom0_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      tok_geom1_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      tok_htopo0_(esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      tok_htopo1_(esConsumes<HcalTopology, HcalRecNumberingRecord>()),
      tok_magField_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      tok_chan_(esConsumes<EcalChannelStatus, EcalChannelStatusRcd>()),
      tok_sevlv_(esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>()),
      tok_topo_(esConsumes<CaloTopology, CaloTopologyRecord>()),
      tok_dbservice_(esConsumes<HcalDbService, HcalDbRecord>()) {
  //now do what ever initialization is needed
  edm::LogVerbatim("HBHEMuon") << "Labels used: Trig " << triggerResults_ << " Vtx " << labelVtx_ << " EB "
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
  edm::LogVerbatim("HBHEMuon") << "Flags used: ollapseDepth " << collapseDepth_ << ":" << mergedDepth_ << " IsItPlan1 "
                               << isItPlan1_ << " IsItPreRecHit " << isItPreRecHit_ << " UseMyCorr " << useMyCorr_;

  //create the objects for HcalHBHEMuonVariables which has information of isolated muons
  produces<HcalHBHEMuonVariablesCollection>(labelHBHEMuon_);
  edm::LogVerbatim("HcalIsoTrack") << " Expected to produce the collections:\n"
                                   << "HcalHBHEMuonVariablesCollection with label " << labelHBHEMuon_;
}

//
// member functions
//

// ------------ method called for each event  ------------
void AlCaHcalHBHEMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++nAll_;
  auto outputHcalHBHEMuonColl = std::make_unique<HcalHBHEMuonVariablesCollection>();

  unsigned int runNumber = iEvent.id().run();
  unsigned int eventNumber = iEvent.id().event();
  unsigned int lumiNumber = iEvent.id().luminosityBlock();
  unsigned int bxNumber = iEvent.bunchCrossing();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber << " Event " << eventNumber << " Lumi " << lumiNumber << " BX "
                               << bxNumber << std::endl;
#endif

  //Step1: Find if the event passes one of the chosen triggers
  bool ok(false);
  /////////////////////////////TriggerResults
  if (trigNames_.empty()) {
    ok = true;
  } else {
    auto const& triggerResults = iEvent.getHandle(tok_trigRes_);
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
        int hlt = triggerResults->accept(iHLT);
        for (unsigned int i = 0; i < trigNames_.size(); ++i) {
          if (triggerNames_[iHLT].find(trigNames_[i]) != std::string::npos) {
            if (hlt > 0) {
              ok = true;
            }
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HBHEMuon") << "AlCaHcalHBHEMuonFilter::Trigger " << triggerNames_[iHLT] << " Flag " << hlt
                                         << ":" << ok << std::endl;
#endif
          }
        }
      }
    }
  }

  // get handles to calogeometry and calotopology
  const HcalDDDRecConstants* hdc = &iSetup.getData(tok_ddrec1_);
  const HcalRespCorrs* resp = &iSetup.getData(tok_respcorr1_);
  const CaloGeometry* geo = &iSetup.getData(tok_geom1_);
  const HcalTopology* theHBHETopology = &iSetup.getData(tok_htopo1_);
  const MagneticField* bField = &iSetup.getData(tok_magField_);
  const EcalChannelStatus* theEcalChStatus = &iSetup.getData(tok_chan_);
  const EcalSeverityLevelAlgo* sevlv = &iSetup.getData(tok_sevlv_);
  const CaloTopology* caloTopology = &iSetup.getData(tok_topo_);
  const HcalDbService* conditions = &iSetup.getData(tok_dbservice_);
  HcalRespCorrs respCorrsObj(*resp);
  HcalRespCorrs* respCorrs = &respCorrsObj;
  respCorrs->setTopo(theHBHETopology);

  // Relevant blocks from iEvent
  auto const& vtx = iEvent.getHandle(tok_Vtx_);
  auto barrelRecHitsHandle = iEvent.getHandle(tok_EB_);
  auto endcapRecHitsHandle = iEvent.getHandle(tok_EE_);
  auto hbhe = iEvent.getHandle(tok_HBHE_);
  auto const& muons = iEvent.getHandle(tok_Muon_);

  // require a good vertex
  math::XYZPoint pvx;
  unsigned int goodVertex = 0;
  reco::VertexCollection::const_iterator firstGoodVertex;
  if (!vtx.isValid()) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "No Good Vertex found == Reject";
#endif
  } else {
    firstGoodVertex = vtx->end();
    for (reco::VertexCollection::const_iterator it = vtx->begin(); it != vtx->end(); it++) {
      if (isGoodVertex(*it)) {
        if (firstGoodVertex == vtx->end())
          firstGoodVertex = it;
        ++goodVertex;
      }
    }
    if (firstGoodVertex != vtx->end())
      pvx = firstGoodVertex->position();
  }

  if (ok && (goodVertex > 0) && muons.isValid() && barrelRecHitsHandle.isValid() && endcapRecHitsHandle.isValid() &&
      hbhe.isValid()) {
    for (reco::MuonCollection::const_iterator recMuon = muons->begin(); recMuon != muons->end(); ++recMuon) {
      HcalHBHEMuonVariables hbheMuon;
      hbheMuon.runNumber_ = runNumber;
      hbheMuon.eventNumber_ = eventNumber;
      hbheMuon.lumiNumber_ = lumiNumber;
      hbheMuon.bxNumber_ = bxNumber;
      hbheMuon.goodVertex_ = goodVertex;
      hbheMuon.muonGood_ = (recMuon->isPFMuon());
      hbheMuon.muonGlobal_ = (recMuon->isGlobalMuon());
      hbheMuon.muonTracker_ = (recMuon->isTrackerMuon());
      hbheMuon.ptGlob_ = ((recMuon)->pt());
      hbheMuon.etaGlob_ = (recMuon->eta());
      hbheMuon.phiGlob_ = (recMuon->phi());
      hbheMuon.energyMuon_ = (recMuon->energy());
      hbheMuon.pMuon_ = (recMuon->p());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HBHEMuon") << "Energy:" << recMuon->energy() << " P:" << recMuon->p();
#endif
      hbheMuon.muonTight_ = (muon::isTightMuon(*recMuon, *firstGoodVertex));
      hbheMuon.muonMedium_ = (muon::isMediumMuon(*recMuon));
      hbheMuon.muonTrkKink_ = (recMuon->combinedQuality().trkKink);
      hbheMuon.muonChi2LocalPosition_ = (recMuon->combinedQuality().chi2LocalPosition);
      hbheMuon.muonSegComp_ = (muon::segmentCompatibility(*recMuon));
      // acessing tracker hits info
      if (recMuon->track().isNonnull()) {
        hbheMuon.trackerLayer_ = (recMuon->track()->hitPattern().trackerLayersWithMeasurement());
      } else {
        hbheMuon.trackerLayer_ = -1;
      }
      if (recMuon->innerTrack().isNonnull()) {
        hbheMuon.innerTrack_ = true;
        hbheMuon.numPixelLayers_ = (recMuon->innerTrack()->hitPattern().pixelLayersWithMeasurement());
        hbheMuon.chiTracker_ = (recMuon->innerTrack()->normalizedChi2());
        hbheMuon.dxyTracker_ = (fabs(recMuon->innerTrack()->dxy(pvx)));
        hbheMuon.dzTracker_ = (fabs(recMuon->innerTrack()->dz(pvx)));
        hbheMuon.innerTrackPt_ = (recMuon->innerTrack()->pt());
        hbheMuon.innerTrackEta_ = (recMuon->innerTrack()->eta());
        hbheMuon.innerTrackPhi_ = (recMuon->innerTrack()->phi());
        hbheMuon.tightPixelHits_ = (recMuon->innerTrack()->hitPattern().numberOfValidPixelHits());
        hbheMuon.tightValidFraction_ = (recMuon->innerTrack()->validFraction());
      }
      // outer track info
      if (recMuon->outerTrack().isNonnull()) {
        hbheMuon.outerTrack_ = true;
        hbheMuon.outerTrackPt_ = (recMuon->outerTrack()->pt());
        hbheMuon.outerTrackEta_ = (recMuon->outerTrack()->eta());
        hbheMuon.outerTrackPhi_ = (recMuon->outerTrack()->phi());
        hbheMuon.outerTrackChi_ = (recMuon->outerTrack()->normalizedChi2());
        hbheMuon.outerTrackHits_ = (recMuon->outerTrack()->numberOfValidHits());
        hbheMuon.outerTrackRHits_ = (recMuon->outerTrack()->recHitsSize());
      }
      // Tight Muon cuts
      if (recMuon->globalTrack().isNonnull()) {
        hbheMuon.globalTrack_ = true;
        hbheMuon.chiGlobal_ = (recMuon->globalTrack()->normalizedChi2());
        hbheMuon.globalMuonHits_ = (recMuon->globalTrack()->hitPattern().numberOfValidMuonHits());
        hbheMuon.matchedStat_ = (recMuon->numberOfMatchedStations());
        hbheMuon.globalTrackPt_ = (recMuon->globalTrack()->pt());
        hbheMuon.globalTrackEta_ = (recMuon->globalTrack()->eta());
        hbheMuon.globalTrackPhi_ = (recMuon->globalTrack()->phi());
        hbheMuon.tightTransImpara_ = (fabs(recMuon->muonBestTrack()->dxy(pvx)));
        hbheMuon.tightLongPara_ = (fabs(recMuon->muonBestTrack()->dz(pvx)));
      }

      hbheMuon.isolationR04_ =
          ((recMuon->pfIsolationR04().sumChargedHadronPt +
            std::max(0.,
                     recMuon->pfIsolationR04().sumNeutralHadronEt + recMuon->pfIsolationR04().sumPhotonEt -
                         (0.5 * recMuon->pfIsolationR04().sumPUPt))) /
           recMuon->pt());

      hbheMuon.isolationR03_ =
          ((recMuon->pfIsolationR03().sumChargedHadronPt +
            std::max(0.,
                     recMuon->pfIsolationR03().sumNeutralHadronEt + recMuon->pfIsolationR03().sumPhotonEt -
                         (0.5 * recMuon->pfIsolationR03().sumPUPt))) /
           recMuon->pt());

      hbheMuon.ecalEnergy_ = (recMuon->calEnergy().emS9);
      hbheMuon.hcalEnergy_ = (recMuon->calEnergy().hadS9);
      hbheMuon.hoEnergy_ = (recMuon->calEnergy().hoS9);

      if (recMuon->innerTrack().isNonnull()) {
        const reco::Track* pTrack = (recMuon->innerTrack()).get();
        spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, (((verbosity_ / 100) % 10 > 0)));

        double activeLengthTot(0), activeLengthHotTot(0);
        double eHcalDepth[depthMax_], eHcalDepthHot[depthMax_];
        double eHcalDepthC[depthMax_], eHcalDepthHotC[depthMax_];
        double cHcalDepthHot[depthMax_], cHcalDepthHotBG[depthMax_];
        double eHcalDepthRaw[depthMax_], eHcalDepthHotRaw[depthMax_];
        double eHcalDepthCRaw[depthMax_], eHcalDepthHotCRaw[depthMax_];
        double cHcalDepthHotRaw[depthMax_], cHcalDepthHotBGRaw[depthMax_];
        double eHcalDepthAux[depthMax_], eHcalDepthHotAux[depthMax_];
        double eHcalDepthCAux[depthMax_], eHcalDepthHotCAux[depthMax_];
        double cHcalDepthHotAux[depthMax_], cHcalDepthHotBGAux[depthMax_];
        double activeL[depthMax_], activeHotL[depthMax_];
        bool matchDepth[depthMax_], matchDepthHot[depthMax_];
        HcalDetId eHcalDetId[depthMax_];
        unsigned int isHot(0);
        int ieta(-1000), iphi(-1000);
        for (int i = 0; i < depthMax_; ++i) {
          eHcalDepth[i] = eHcalDepthHot[i] = 0;
          eHcalDepthC[i] = eHcalDepthHotC[i] = 0;
          cHcalDepthHot[i] = cHcalDepthHotBG[i] = 0;
          eHcalDepthRaw[i] = eHcalDepthHotRaw[i] = 0;
          eHcalDepthCRaw[i] = eHcalDepthHotCRaw[i] = 0;
          cHcalDepthHotRaw[i] = cHcalDepthHotBGRaw[i] = 0;
          eHcalDepthAux[i] = eHcalDepthHotAux[i] = 0;
          eHcalDepthCAux[i] = eHcalDepthHotCAux[i] = 0;
          cHcalDepthHotAux[i] = cHcalDepthHotBGAux[i] = 0;
          activeL[i] = activeHotL[i] = 0;
          matchDepth[i] = matchDepthHot[i] = true;
        }
#ifdef EDM_ML_DEBUG
        double eHcal(0);
#endif

        hbheMuon.ecalDetId_ = ((trackID.detIdECAL)());
        hbheMuon.hcalDetId_ = ((trackID.detIdHCAL)());
        hbheMuon.ehcalDetId_ = ((trackID.detIdEHCAL)());

        HcalDetId check(false);
        std::pair<bool, HcalDetId> info = spr::propagateHCALBack(pTrack, geo, bField, (((verbosity_ / 100) % 10 > 0)));
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
                                                          geo,
                                                          caloTopology,
                                                          sevlv,
                                                          1,
                                                          1,
                                                          -100.0,
                                                          -100.0,
                                                          -500.0,
                                                          500.0,
                                                          false);
          hbheMuon.ecal3x3Energy_ = e3x3.first;
          okE = e3x3.second;
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HBHEMuon") << "Propagate Track to ECAL: " << okE << ":" << trackID.okECAL << " E "
                                     << hbheMuon.ecal3x3Energy_;
#endif

        if (trackID.okHCAL) {
          DetId closestCell(trackID.detIdHCAL);
          HcalDetId hcidt(closestCell.rawId());
          if ((hcidt.ieta() == check.ieta()) && (hcidt.iphi() == check.iphi()))
            hbheMuon.matchedId_ = true;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HBHEMuon") << "Front " << hcidt << " Back " << info.first << ":" << check << " Match "
                                       << hbheMuon.matchedId_;
#endif
          HcalSubdetector subdet = hcidt.subdet();
          ieta = hcidt.ieta();
          iphi = hcidt.iphi();
          bool hborhe = (std::abs(ieta) == 16);

          hbheMuon.hcal1x1Energy_ = spr::eHCALmatrix(
              theHBHETopology, closestCell, hbhe, 0, 0, false, true, -100.0, -100.0, -100.0, -100.0, -500., 500., 0);
          hbheMuon.hcal1x1EnergyAux_ = spr::eHCALmatrix(
              theHBHETopology, closestCell, hbhe, 0, 0, false, true, -100.0, -100.0, -100.0, -100.0, -500., 500., 1);
          hbheMuon.hcal1x1EnergyRaw_ = spr::eHCALmatrix(
              theHBHETopology, closestCell, hbhe, 0, 0, false, true, -100.0, -100.0, -100.0, -100.0, -500., 500., 2);
          std::vector<std::pair<double, int>> ehdepth, ehdepthAux, ehdepthRaw;
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
                              0,
                              depth16HE(ieta, iphi, theHBHETopology),
                              (((verbosity_ / 1000) % 10) > 0));
          for (int i = 0; i < depthMax_; ++i)
            eHcalDetId[i] = HcalDetId();
          spr::energyHCALCell((HcalDetId)closestCell,
                              hbhe,
                              ehdepthAux,
                              maxDepth_,
                              -100.0,
                              -100.0,
                              -100.0,
                              -100.0,
                              -500.0,
                              500.0,
                              1,
                              depth16HE(ieta, iphi, theHBHETopology),
                              (((verbosity_ / 1000) % 10) > 0));
          spr::energyHCALCell((HcalDetId)closestCell,
                              hbhe,
                              ehdepthRaw,
                              maxDepth_,
                              -100.0,
                              -100.0,
                              -100.0,
                              -100.0,
                              -500.0,
                              500.0,
                              2,
                              depth16HE(ieta, iphi, theHBHETopology),
                              (((verbosity_ / 1000) % 10) > 0));
          for (unsigned int i = 0; i < ehdepth.size(); ++i) {
            HcalSubdetector subdet0 =
                (hborhe) ? ((ehdepth[i].second >= depth16HE(ieta, iphi, theHBHETopology)) ? HcalEndcap : HcalBarrel)
                         : subdet;
            HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
            double actL = activeLength(DetId(hcid0), hdc);
            double ene = ehdepth[i].first;
            double eneAux = ehdepthAux[i].first;
            double eneRaw = ehdepthRaw[i].first;
            bool tmpC(false);
            if (ene > 0.0) {
              if (!(theHBHETopology->validHcal(hcid0))) {
                edm::LogWarning("HBHEMuon") << "(1) Invalid ID " << hcid0 << " with E = " << ene;
                edm::LogWarning("HBHEMuon") << HcalDetId(closestCell) << " with " << ehdepth.size() << " depths:";
                for (const auto& ehd : ehdepth)
                  edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
              } else {
                tmpC = goodCell(hcid0, pTrack, geo, bField, hdc);
                double enec(ene);
                double corr = respCorr(DetId(hcid0), respCorrs);
                if (corr != 0)
                  ene /= corr;
#ifdef EDM_ML_DEBUG
                HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc->mergedDepthDetId(hcid0) : hcid0;
                edm::LogVerbatim("HBHEMuon") << hcid0 << ":" << id << " Corr " << corr;
#endif
                int depth = ehdepth[i].second - 1;
                if (collapseDepth_) {
                  HcalDetId id = hdc->mergedDepthDetId(hcid0);
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
            if (eneAux > 0.0) {
              if (theHBHETopology->validHcal(hcid0)) {
                double enecAux(eneAux);
                double corr = respCorr(DetId(hcid0), respCorrs);
                if (corr != 0)
                  eneAux /= corr;
                int depth = ehdepthAux[i].second - 1;
                if (collapseDepth_) {
                  HcalDetId id = hdc->mergedDepthDetId(hcid0);
                  depth = id.depth() - 1;
                }
                eHcalDepthAux[depth] += eneAux;
                eHcalDepthCAux[depth] += enecAux;
#ifdef EDM_ML_DEBUG
                if ((verbosity_ % 10) > 0)
                  edm::LogVerbatim("HBHEMuon")
                      << hcid0 << " E " << eneAux << ":" << enecAux << " L " << actL << " Match " << tmpC;
#endif
              }
            }
            if (eneRaw > 0.0) {
              if (theHBHETopology->validHcal(hcid0)) {
                double enecRaw(eneRaw);
                double corr = respCorr(DetId(hcid0), respCorrs);
                if (corr != 0)
                  eneRaw /= corr;
                int depth = ehdepthRaw[i].second - 1;
                if (collapseDepth_) {
                  HcalDetId id = hdc->mergedDepthDetId(hcid0);
                  depth = id.depth() - 1;
                }
                eHcalDepthRaw[depth] += eneRaw;
                eHcalDepthCRaw[depth] += enecRaw;
#ifdef EDM_ML_DEBUG
                if ((verbosity_ % 10) > 0)
                  edm::LogVerbatim("HBHEMuon")
                      << hcid0 << " E " << eneRaw << ":" << enecRaw << " L " << actL << " Match " << tmpC;
#endif
              }
            }
          }
#ifdef EDM_ML_DEBUG
          if ((verbosity_ % 10) > 0) {
            edm::LogVerbatim("HBHEMuon") << hcidt << " Match " << hbheMuon.matchedId_ << " Depths " << ehdepth.size();
            for (unsigned int k = 0; k < ehdepth.size(); ++k)
              edm::LogVerbatim("HBHEMuon") << " [" << k << ":" << ehdepth[k].second << "] " << matchDepth[k];
          }
#endif
          HcalDetId hotCell;
          spr::eHCALmatrix(geo, theHBHETopology, closestCell, hbhe, 1, 1, hotCell, false, 0, false);
          isHot = matchId(closestCell, hotCell);
          if (hotCell != HcalDetId()) {
            subdet = HcalDetId(hotCell).subdet();
            ieta = HcalDetId(hotCell).ieta();
            iphi = HcalDetId(hotCell).iphi();
            hborhe = (std::abs(ieta) == 16);
            std::vector<std::pair<double, int>> ehdepth, ehdepthAux, ehdepthRaw;
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
                                0,
                                depth16HE(ieta, iphi, theHBHETopology),
                                false);
            spr::energyHCALCell(hotCell,
                                hbhe,
                                ehdepthAux,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                1,
                                depth16HE(ieta, iphi, theHBHETopology),
                                false);
            spr::energyHCALCell(hotCell,
                                hbhe,
                                ehdepthRaw,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                2,
                                depth16HE(ieta, iphi, theHBHETopology),
                                false);
            for (int i = 0; i < depthMax_; ++i)
              eHcalDetId[i] = HcalDetId();
            for (unsigned int i = 0; i < ehdepth.size(); ++i) {
              HcalSubdetector subdet0 =
                  (hborhe) ? ((ehdepth[i].second >= depth16HE(ieta, iphi, theHBHETopology)) ? HcalEndcap : HcalBarrel)
                           : subdet;
              HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
              double actL = activeLength(DetId(hcid0), hdc);
              double ene = ehdepth[i].first;
              bool tmpC(false);
              if (ene > 0.0) {
                if (!(theHBHETopology->validHcal(hcid0))) {
                  edm::LogWarning("HBHEMuon") << "(2) Invalid ID " << hcid0 << " with E = " << ene;
                  edm::LogWarning("HBHEMuon") << HcalDetId(hotCell) << " with " << ehdepth.size() << " depths:";
                  for (const auto& ehd : ehdepth)
                    edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
                } else {
                  tmpC = goodCell(hcid0, pTrack, geo, bField, hdc);
                  double chg(ene), enec(ene);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    ene /= corr;
#ifdef EDM_ML_DEBUG
                  HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc->mergedDepthDetId(hcid0) : hcid0;
                  edm::LogVerbatim("HBHEMuon") << hcid0 << ":" << id << " Corr " << corr << " E " << ene << ":" << enec;
#endif
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chg /= gain;
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HBHEMuon") << hcid0 << " Gain " << gain << " C " << chg;
#endif
                  int depth = ehdepth[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  eHcalDepthHot[depth] += ene;
                  eHcalDepthHotC[depth] += enec;
                  cHcalDepthHot[depth] += chg;
                  activeHotL[depth] += actL;
                  activeLengthHotTot += actL;
                  matchDepthHot[depth] = (matchDepthHot[depth] && tmpC);
#ifdef EDM_ML_DEBUG
                  eHcal += ene;
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " depth " << depth << " E " << ene << ":" << enec << " C "
                                                 << chg << " L " << actL << " Match " << tmpC;
#endif
                }
              }
              double eneAux = ehdepthAux[i].first;
              if (eneAux > 0.0) {
                if (theHBHETopology->validHcal(hcid0)) {
                  double chgAux(eneAux), enecAux(eneAux);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    eneAux /= corr;
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chgAux /= gain;
                  int depth = ehdepthAux[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  eHcalDepthHotAux[depth] += eneAux;
                  eHcalDepthHotCAux[depth] += enecAux;
                  cHcalDepthHotAux[depth] += chgAux;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " depth " << depth << " E " << eneAux << ":" << enecAux
                                                 << " C " << chgAux << " L " << actL << " Match " << tmpC;
#endif
                }
              }
              double eneRaw = ehdepthRaw[i].first;
              if (eneRaw > 0.0) {
                if (theHBHETopology->validHcal(hcid0)) {
                  double chgRaw(eneRaw), enecRaw(eneRaw);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    eneRaw /= corr;
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chgRaw /= gain;
                  int depth = ehdepthRaw[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  eHcalDepthHotRaw[depth] += eneRaw;
                  eHcalDepthHotCRaw[depth] += enecRaw;
                  cHcalDepthHotRaw[depth] += chgRaw;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " depth " << depth << " E " << eneRaw << ":" << enecRaw
                                                 << " C " << chgRaw << " L " << actL << " Match " << tmpC;
#endif
                }
              }
            }

            HcalDetId oppCell(subdet, -ieta, iphi, HcalDetId(hotCell).depth());
            std::vector<std::pair<double, int>> ehdeptho, ehdepthoAux, ehdepthoRaw;
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
                                0,
                                depth16HE(-ieta, iphi, theHBHETopology),
                                false);
            spr::energyHCALCell(oppCell,
                                hbhe,
                                ehdepthoAux,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                1,
                                depth16HE(-ieta, iphi, theHBHETopology),
                                false);
            spr::energyHCALCell(oppCell,
                                hbhe,
                                ehdepthoRaw,
                                maxDepth_,
                                -100.0,
                                -100.0,
                                -100.0,
                                -100.0,
                                -500.0,
                                500.0,
                                2,
                                depth16HE(-ieta, iphi, theHBHETopology),
                                false);
            for (unsigned int i = 0; i < ehdeptho.size(); ++i) {
              HcalSubdetector subdet0 =
                  (hborhe) ? ((ehdeptho[i].second >= depth16HE(-ieta, iphi, theHBHETopology)) ? HcalEndcap : HcalBarrel)
                           : subdet;
              HcalDetId hcid0(subdet0, -ieta, iphi, ehdeptho[i].second);
              double ene = ehdeptho[i].first;
              if (ene > 0.0) {
                if (!(theHBHETopology->validHcal(hcid0))) {
                  edm::LogWarning("HBHEMuon") << "(3) Invalid ID " << hcid0 << " with E = " << ene;
                  edm::LogWarning("HBHEMuon") << oppCell << " with " << ehdeptho.size() << " depths:";
                  for (const auto& ehd : ehdeptho)
                    edm::LogWarning("HBHEMuon") << " " << ehd.second << ":" << ehd.first;
                } else {
                  double chg(ene);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    ene /= corr;
#ifdef EDM_ML_DEBUG
                  HcalDetId id = (isItPlan1_ && isItPreRecHit_) ? hdc->mergedDepthDetId(hcid0) : hcid0;
                  edm::LogVerbatim("HBHEMuon")
                      << hcid0 << ":" << id << " Corr " << corr << " E " << ene << ":" << ehdeptho[i].first;
#endif
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chg /= gain;
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HBHEMuon") << hcid0 << " Gain " << gain << " C " << chg;
#endif
                  int depth = ehdeptho[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  cHcalDepthHotBG[depth] += chg;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Depth " << depth << " E " << ene << " C " << chg;
#endif
                }
              }
              double eneAux = ehdepthoAux[i].first;
              if (eneAux > 0.0) {
                if (theHBHETopology->validHcal(hcid0)) {
                  double chgAux(eneAux);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    eneAux /= corr;
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chgAux /= gain;
                  int depth = ehdepthoAux[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  cHcalDepthHotBGAux[depth] += chgAux;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Depth " << depth << " E " << eneAux << " C " << chgAux;
#endif
                }
              }
              double eneRaw = ehdepthoRaw[i].first;
              if (eneRaw > 0.0) {
                if (theHBHETopology->validHcal(hcid0)) {
                  double chgRaw(eneRaw);
                  double corr = respCorr(DetId(hcid0), respCorrs);
                  if (corr != 0)
                    eneRaw /= corr;
                  double gain = gainFactor(conditions, hcid0);
                  if (gain != 0)
                    chgRaw /= gain;
                  int depth = ehdepthoRaw[i].second - 1;
                  if (collapseDepth_) {
                    HcalDetId id = hdc->mergedDepthDetId(hcid0);
                    depth = id.depth() - 1;
                  }
                  cHcalDepthHotBGRaw[depth] += chgRaw;
#ifdef EDM_ML_DEBUG
                  if ((verbosity_ % 10) > 0)
                    edm::LogVerbatim("HBHEMuon") << hcid0 << " Depth " << depth << " E " << eneRaw << " C " << chgRaw;
#endif
                }
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HBHEMuon") << "Propagate Track to HCAL: " << trackID.okHCAL << " Match "
                                     << hbheMuon.matchedId_ << " Hot " << isHot << " Energy " << eHcal;
#endif
        hbheMuon.hcalIeta_ = ieta;
        hbheMuon.hcalIphi_ = iphi;
        for (int i = 0; i < depthMax_; ++i) {
          hbheMuon.hcalDepthEnergy_.push_back(eHcalDepth[i]);
          hbheMuon.hcalDepthActiveLength_.push_back(activeL[i]);
          hbheMuon.hcalDepthEnergyHot_.push_back(eHcalDepthHot[i]);
          hbheMuon.hcalDepthActiveLengthHot_.push_back(activeHotL[i]);
          hbheMuon.hcalDepthEnergyCorr_.push_back(eHcalDepthC[i]);
          hbheMuon.hcalDepthEnergyHotCorr_.push_back(eHcalDepthHotC[i]);
          hbheMuon.hcalDepthChargeHot_.push_back(cHcalDepthHot[i]);
          hbheMuon.hcalDepthChargeHotBG_.push_back(cHcalDepthHotBG[i]);
          hbheMuon.hcalDepthMatch_.push_back(matchDepth[i]);
          hbheMuon.hcalDepthMatchHot_.push_back(matchDepthHot[i]);
          hbheMuon.hcalDepthEnergyAux_.push_back(eHcalDepthAux[i]);
          hbheMuon.hcalDepthEnergyHotAux_.push_back(eHcalDepthHotAux[i]);
          hbheMuon.hcalDepthEnergyCorrAux_.push_back(eHcalDepthCAux[i]);
          hbheMuon.hcalDepthEnergyHotCorrAux_.push_back(eHcalDepthHotCAux[i]);
          hbheMuon.hcalDepthChargeHotAux_.push_back(cHcalDepthHotAux[i]);
          hbheMuon.hcalDepthChargeHotBGAux_.push_back(cHcalDepthHotBGAux[i]);
          hbheMuon.hcalDepthEnergyRaw_.push_back(eHcalDepthRaw[i]);
          hbheMuon.hcalDepthEnergyHotRaw_.push_back(eHcalDepthHotRaw[i]);
          hbheMuon.hcalDepthEnergyCorrRaw_.push_back(eHcalDepthCRaw[i]);
          hbheMuon.hcalDepthEnergyHotCorrRaw_.push_back(eHcalDepthHotCRaw[i]);
          hbheMuon.hcalDepthChargeHotRaw_.push_back(cHcalDepthHotRaw[i]);
          hbheMuon.hcalDepthChargeHotBGRaw_.push_back(cHcalDepthHotBGRaw[i]);
        }
        hbheMuon.hcalActiveLength_ = activeLengthTot;
        hbheMuon.hcalHot_ = isHot;
        hbheMuon.hcalActiveLengthHot_ = activeLengthHotTot;

        if ((recMuon->p() > 10.0) && (trackID.okHCAL))
          outputHcalHBHEMuonColl->emplace_back(hbheMuon);
      }
    }
  }
  if (!outputHcalHBHEMuonColl->empty())
    ++nGood_;
  iEvent.put(std::move(outputHcalHBHEMuonColl), labelHBHEMuon_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaHcalHBHEMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> triggers = {"HLT_IsoMu", "HLT_Mu"};
  desc.add<std::vector<std::string>>("triggers", triggers);
  desc.add<std::string>("processName", "HLT");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("labelEBRecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("labelEERecHit", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("labelHBHERecHit", edm::InputTag("hbhereco"));
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelMuon", "muons");
  desc.add<std::string>("labelHBHEMuon", "hbheMuon");
  desc.add<bool>("collapseDepth", false);
  desc.add<bool>("isItPlan1", false);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<bool>("isItPreRecHit", false);
  desc.addUntracked<bool>("writeRespCorr", false);
  desc.addUntracked<std::string>("fileInCorr", "");
  desc.addUntracked<int>("maxDepth", 4);
  descriptions.add("alcaHcalHBHEMuonProducer", desc);
}

// ------------ method called once each job just after ending the event loop  ------------
void AlCaHcalHBHEMuonProducer::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaHcalHBHEMuonProducer::globalEndJob(const alcaHcalHBHEMuon::Counters* count) {
  edm::LogVerbatim("HBHEMuon") << "Selects " << count->nGood_ << " out of " << count->nAll_ << " total # of events\n";
}

// ------------ method called when starting or ending a run  ------------
void AlCaHcalHBHEMuonProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  const HcalDDDRecConstants* hdc = &iSetup.getData(tok_ddrec0_);
  actHB.clear();
  actHE.clear();
  actHB = hdc->getThickActive(0);
  actHE = hdc->getThickActive(1);
#ifdef EDM_ML_DEBUG
  unsigned int k1(0), k2(0);
  edm::LogVerbatim("HBHEMuon") << actHB.size() << " Active Length for HB";
  for (const auto& act : actHB) {
    edm::LogVerbatim("HBHEMuon") << "[" << k1 << "] ieta " << act.ieta << " depth " << act.depth << " zside "
                                 << act.zside << " type " << act.stype << " phi " << act.iphis.size() << ":"
                                 << act.iphis[0] << " L " << act.thick;
    HcalDetId hcid1(HcalBarrel, (act.ieta) * (act.zside), act.iphis[0], act.depth);
    HcalDetId hcid2 = mergedDepth_ ? hdc->mergedDepthDetId(hcid1) : hcid1;
    edm::LogVerbatim("HBHEMuon") << hcid1 << " | " << hcid2 << " L " << activeLength(DetId(hcid2), hdc);
    ++k1;
  }
  edm::LogVerbatim("HBHEMuon") << actHE.size() << " Active Length for HE";
  for (const auto& act : actHE) {
    edm::LogVerbatim("HBHEMuon") << "[" << k2 << "] ieta " << act.ieta << " depth " << act.depth << " zside "
                                 << act.zside << " type " << act.stype << " phi " << act.iphis.size() << ":"
                                 << act.iphis[0] << " L " << act.thick;
    HcalDetId hcid1(HcalEndcap, (act.ieta) * (act.zside), act.iphis[0], act.depth);
    HcalDetId hcid2 = mergedDepth_ ? hdc->mergedDepthDetId(hcid1) : hcid1;
    edm::LogVerbatim("HBHEMuon") << hcid1 << " | " << hcid2 << " L " << activeLength(DetId(hcid2), hdc);
    ++k2;
  }
#endif

  bool changed = true;
  bool flag = hltConfig_.init(iRun, iSetup, processName_, changed);
  edm::LogVerbatim("HBHEMuon") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init " << flag << std::endl;

  const HcalRespCorrs* resp = &iSetup.getData(tok_respcorr0_);
  const HcalTopology* theHBHETopology = &iSetup.getData(tok_htopo0_);
  const CaloGeometry* geo = &iSetup.getData(tok_geom0_);
  HcalRespCorrs respCorrsObj(*resp);
  HcalRespCorrs* respCorrs = &respCorrsObj;
  respCorrs->setTopo(theHBHETopology);

  // Write correction factors for all HB/HE events
  if (writeRespCorr_) {
    const HcalGeometry* gHcal = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
    const std::vector<DetId>& ids = gHcal->getValidDetIds(DetId::Hcal, 0);
    edm::LogVerbatim("HBHEMuon") << "\nTable of Correction Factors for Run " << iRun.run() << "\n";
    for (auto const& id : ids) {
      if ((id.det() == DetId::Hcal) && ((id.subdetId() == HcalBarrel) || (id.subdetId() == HcalEndcap))) {
        edm::LogVerbatim("HBHEMuon") << HcalDetId(id) << " " << id.rawId() << " "
                                     << (respCorrs->getValues(id))->getValue();
      }
    }
  }
}

void AlCaHcalHBHEMuonProducer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogVerbatim("HBHEMuon") << "endRun[" << nRun_ << "] " << iRun.run() << "\n";
  ++nRun_;
}

// ------------ methods called by produce()  ------------

int AlCaHcalHBHEMuonProducer::matchId(const HcalDetId& id1, const HcalDetId& id2) {
  HcalDetId kd1(id1.subdet(), id1.ieta(), id1.iphi(), 1);
  HcalDetId kd2(id1.subdet(), id2.ieta(), id2.iphi(), 1);
  int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double AlCaHcalHBHEMuonProducer::activeLength(const DetId& id0, const HcalDDDRecConstants* hdc) {
  HcalDetId id(id0);
  int ieta = id.ietaAbs();
  int zside = id.zside();
  int iphi = id.iphi();
  std::vector<int> dpths;
  if (mergedDepth_) {
    std::vector<HcalDetId> ids;
    hdc->unmergeDepthDetId(id, ids);
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

bool AlCaHcalHBHEMuonProducer::isGoodVertex(const reco::Vertex& vtx) {
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

double AlCaHcalHBHEMuonProducer::respCorr(const DetId& id, const HcalRespCorrs* respCorrs) {
  double cfac(1.0);
  if (useMyCorr_) {
    auto itr = corrValue_.find(id);
    if (itr != corrValue_.end())
      cfac = itr->second;
  } else if (respCorrs != nullptr) {
    cfac = (respCorrs->getValues(id))->getValue();
  }
  return cfac;
}

double AlCaHcalHBHEMuonProducer::gainFactor(const HcalDbService* conditions, const HcalDetId& id) {
  double gain(0.0);
  const HcalCalibrations& calibs = conditions->getHcalCalibrations(id);
  for (int capid = 0; capid < 4; ++capid)
    gain += (0.25 * calibs.respcorrgain(capid));
  return gain;
}

int AlCaHcalHBHEMuonProducer::depth16HE(int ieta, int iphi, const HcalTopology* theHBHETopology) {
  // Transition between HB/HE is special
  // For Run 1 or for Plan1 standard reconstruction it is 3
  // For runs beyond 2018 or in Plan1 for HEP17 it is 4
  int zside = (ieta > 0) ? 1 : -1;
  int depth = theHBHETopology->dddConstants()->getMinDepth(1, 16, iphi, zside);
  if (isItPlan1_ && (!isItPreRecHit_))
    depth = 3;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Plan1 " << isItPlan1_ << " PreRecHit " << isItPreRecHit_ << " phi " << iphi
                               << " depth " << depth;
#endif
  return depth;
}

bool AlCaHcalHBHEMuonProducer::goodCell(const HcalDetId& hcid,
                                        const reco::Track* pTrack,
                                        const CaloGeometry* geo,
                                        const MagneticField* bField,
                                        const HcalDDDRecConstants* hdc) {
  std::pair<double, double> rz = hdc->getRZ(hcid);
  bool typeRZ = (hcid.subdet() == HcalEndcap) ? false : true;
  bool match = spr::propagateHCAL(pTrack, geo, bField, typeRZ, rz, (((verbosity_ / 10000) % 10) > 0));
  return match;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHcalHBHEMuonProducer);
