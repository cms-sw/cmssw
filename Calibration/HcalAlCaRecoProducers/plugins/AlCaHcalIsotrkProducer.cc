// -*- C++ -*-

// system include files
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

// user include files
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalCalibObjects/interface/HcalIsoTrkCalibVariables.h"
#include "DataFormats/HcalCalibObjects/interface/HcalIsoTrkEventVariables.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alCaHcalIsotrkProducer {
  struct Counters {
    Counters() : nAll_(0), nGood_(0), nRange_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_, nRange_;
  };
}  // namespace alCaHcalIsotrkProducer

class AlCaHcalIsotrkProducer : public edm::stream::EDProducer<edm::GlobalCache<alCaHcalIsotrkProducer::Counters>> {
public:
  explicit AlCaHcalIsotrkProducer(edm::ParameterSet const&, const alCaHcalIsotrkProducer::Counters*);
  ~AlCaHcalIsotrkProducer() override = default;

  static std::unique_ptr<alCaHcalIsotrkProducer::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<alCaHcalIsotrkProducer::Counters>();
  }

  void produce(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const alCaHcalIsotrkProducer::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  std::array<int, 3> getProducts(int goodPV,
                                 double eventWeight,
                                 std::vector<math::XYZTLorentzVector>& vecL1,
                                 std::vector<math::XYZTLorentzVector>& vecL3,
                                 math::XYZPoint& leadPV,
                                 std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
                                 std::vector<spr::propagatedTrackID>& trkCaloDets,
                                 const CaloGeometry* geo,
                                 const CaloTopology* topo,
                                 const HcalTopology* theHBHETopology,
                                 const EcalChannelStatus* theEcalChStatus,
                                 const EcalSeverityLevelAlgo* theEcalSevlv,
                                 edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
                                 edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
                                 edm::Handle<HBHERecHitCollection>& hbhe,
                                 edm::Handle<CaloTowerCollection>& towerHandle,
                                 const edm::Handle<reco::GenParticleCollection>& genParticles,
                                 const HcalRespCorrs* respCorrs,
                                 const edm::Handle<reco::MuonCollection>& muonh,
                                 std::vector<HcalIsoTrkCalibVariables>& hocalib,
                                 HcalIsoTrkEventVariables& hocalibEvent,
                                 const edm::EventID& eventId);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double trackP(const reco::Track*, const edm::Handle<reco::GenParticleCollection>&);
  double rhoh(const edm::Handle<CaloTowerCollection>&);
  double eThreshold(const DetId& id, const CaloGeometry* geo) const;
  DetId newId(const DetId&);
  void storeEnergy(const HcalRespCorrs* respCorrs,
                   const std::vector<DetId>& ids,
                   std::vector<double>& edet,
                   double& eHcal,
                   std::vector<unsigned int>& detIds,
                   std::vector<double>& hitEnergies);
  std::pair<double, double> storeEnergy(const HcalRespCorrs* respCorrs,
                                        edm::Handle<HBHERecHitCollection>& hbhe,
                                        const std::vector<DetId>& ids,
                                        std::vector<double>& hitEnergy1,
                                        std::vector<double>& hitEnergy2);
  bool notaMuon(const reco::Track* pTrack0, const edm::Handle<reco::MuonCollection>& muonh);

  // ----------member data ---------------------------
  l1t::L1TGlobalUtil* l1GtUtils_;
  HLTConfigProvider hltConfig_;
  unsigned int nRun_, nAll_, nGood_, nRange_;
  const std::vector<std::string> trigNames_;
  spr::trackSelectionParameters selectionParameter_;
  const std::string theTrackQuality_;
  const std::string processName_, l1Filter_;
  const std::string l2Filter_, l3Filter_;
  const double a_coneR_, a_mipR_, a_mipR2_, a_mipR3_;
  const double a_mipR4_, a_mipR5_, pTrackMin_, eEcalMax_;
  const double maxRestrictionP_, slopeRestrictionP_;
  const double hcalScale_, eIsolate1_, eIsolate2_;
  const double pTrackLow_, pTrackHigh_;
  const bool ignoreTrigger_, useL1Trigger_;
  const bool unCorrect_, collapseDepth_;
  const double hitEthrEB_, hitEthrEE0_, hitEthrEE1_;
  const double hitEthrEE2_, hitEthrEE3_;
  const double hitEthrEELo_, hitEthrEEHi_;
  const edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  const std::string labelGenTrack_, labelRecVtx_, labelEB_;
  const std::string labelEE_, labelHBHE_, labelTower_, l1TrigName_;
  const std::vector<int> oldID_, newDepth_;
  const bool hep17_;
  const std::string labelIsoTkVar_, labelIsoTkEvtVar_;
  const std::vector<int> debEvents_;
  const bool usePFThresh_;

  double a_charIsoR_, a_coneR1_, a_coneR2_;
  const HcalDDDRecConstants* hdc_;
  const EcalPFRecHitThresholds* eThresholds_;

  std::vector<double> etabins_, phibins_;
  std::vector<int> oldDet_, oldEta_, oldDepth_;
  double etadist_, phidist_, etahalfdist_, phihalfdist_;

  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<CaloTowerCollection> tok_cala_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;
  edm::EDGetTokenT<BXVector<GlobalAlgBlk>> tok_alg_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_bFieldH_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_ecalChStatus_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_resp_;
  edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> tok_ecalPFRecHitThresholds_;

  bool debug_;
};

AlCaHcalIsotrkProducer::AlCaHcalIsotrkProducer(edm::ParameterSet const& iConfig,
                                               const alCaHcalIsotrkProducer::Counters* counters)
    : nRun_(0),
      nAll_(0),
      nGood_(0),
      nRange_(0),
      trigNames_(iConfig.getParameter<std::vector<std::string>>("triggers")),
      theTrackQuality_(iConfig.getParameter<std::string>("trackQuality")),
      processName_(iConfig.getParameter<std::string>("processName")),
      l1Filter_(iConfig.getParameter<std::string>("l1Filter")),
      l2Filter_(iConfig.getParameter<std::string>("l2Filter")),
      l3Filter_(iConfig.getParameter<std::string>("l3Filter")),
      a_coneR_(iConfig.getParameter<double>("coneRadius")),
      a_mipR_(iConfig.getParameter<double>("coneRadiusMIP")),
      a_mipR2_(iConfig.getParameter<double>("coneRadiusMIP2")),
      a_mipR3_(iConfig.getParameter<double>("coneRadiusMIP3")),
      a_mipR4_(iConfig.getParameter<double>("coneRadiusMIP4")),
      a_mipR5_(iConfig.getParameter<double>("coneRadiusMIP5")),
      pTrackMin_(iConfig.getParameter<double>("minimumTrackP")),
      eEcalMax_(iConfig.getParameter<double>("maximumEcalEnergy")),
      maxRestrictionP_(iConfig.getParameter<double>("maxTrackP")),
      slopeRestrictionP_(iConfig.getParameter<double>("slopeTrackP")),
      hcalScale_(iConfig.getUntrackedParameter<double>("hHcalScale", 1.0)),
      eIsolate1_(iConfig.getParameter<double>("isolationEnergyTight")),
      eIsolate2_(iConfig.getParameter<double>("isolationEnergyLoose")),
      pTrackLow_(iConfig.getParameter<double>("momentumLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumHigh")),
      ignoreTrigger_(iConfig.getUntrackedParameter<bool>("ignoreTriggers", false)),
      useL1Trigger_(iConfig.getUntrackedParameter<bool>("useL1Trigger", false)),
      unCorrect_(iConfig.getUntrackedParameter<bool>("unCorrect", false)),
      collapseDepth_(iConfig.getUntrackedParameter<bool>("collapseDepth", false)),
      hitEthrEB_(iConfig.getParameter<double>("EBHitEnergyThreshold")),
      hitEthrEE0_(iConfig.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(iConfig.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(iConfig.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(iConfig.getParameter<double>("EEHitEnergyThreshold3")),
      hitEthrEELo_(iConfig.getParameter<double>("EEHitEnergyThresholdLow")),
      hitEthrEEHi_(iConfig.getParameter<double>("EEHitEnergyThresholdHigh")),
      triggerEvent_(iConfig.getParameter<edm::InputTag>("labelTriggerEvent")),
      theTriggerResultsLabel_(iConfig.getParameter<edm::InputTag>("labelTriggerResult")),
      labelGenTrack_(iConfig.getParameter<std::string>("labelTrack")),
      labelRecVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelEB_(iConfig.getParameter<std::string>("labelEBRecHit")),
      labelEE_(iConfig.getParameter<std::string>("labelEERecHit")),
      labelHBHE_(iConfig.getParameter<std::string>("labelHBHERecHit")),
      labelTower_(iConfig.getParameter<std::string>("labelCaloTower")),
      l1TrigName_(iConfig.getUntrackedParameter<std::string>("l1TrigName", "L1_SingleJet60")),
      oldID_(iConfig.getUntrackedParameter<std::vector<int>>("oldID")),
      newDepth_(iConfig.getUntrackedParameter<std::vector<int>>("newDepth")),
      hep17_(iConfig.getUntrackedParameter<bool>("hep17")),
      labelIsoTkVar_(iConfig.getParameter<std::string>("isoTrackLabel")),
      labelIsoTkEvtVar_(iConfig.getParameter<std::string>("isoTrackEventLabel")),
      debEvents_(iConfig.getParameter<std::vector<int>>("debugEvents")),
      usePFThresh_(iConfig.getParameter<bool>("usePFThreshold")) {
  // Get the run parameters
  const double isolationRadius(28.9), innerR(10.0), outerR(30.0);
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt = iConfig.getParameter<double>("minTrackPt");
  selectionParameter_.maxChi2 = iConfig.getParameter<double>("maxChi2");
  selectionParameter_.maxDpOverP = iConfig.getParameter<double>("maxDpOverP");
  selectionParameter_.minOuterHit = iConfig.getParameter<int>("minOuterHit");
  selectionParameter_.minLayerCrossed = iConfig.getParameter<int>("minLayerCrossed");
  selectionParameter_.maxInMiss = iConfig.getParameter<int>("maxInMiss");
  selectionParameter_.minQuality = trackQuality_;
  selectionParameter_.maxDxyPV = iConfig.getParameter<double>("maxDxyPV");
  selectionParameter_.maxDzPV = iConfig.getParameter<double>("maxDzPV");
  selectionParameter_.maxChi2 = iConfig.getParameter<double>("maxChi2");
  selectionParameter_.maxDpOverP = iConfig.getParameter<double>("maxDpOverP");
  selectionParameter_.minOuterHit = iConfig.getParameter<int>("minOuterHit");
  selectionParameter_.minLayerCrossed = iConfig.getParameter<int>("minLayerCrossed");
  selectionParameter_.maxInMiss = iConfig.getParameter<int>("maxInMiss");
  selectionParameter_.maxOutMiss = iConfig.getParameter<int>("maxOutMiss");
  a_charIsoR_ = a_coneR_ + isolationRadius;
  a_coneR1_ = a_coneR_ + innerR;
  a_coneR2_ = a_coneR_ + outerR;
  // Different isolation cuts are described in DN-2016/029
  // Tight cut uses 2 GeV; Loose cut uses 10 GeV
  // Eta dependent cut uses (maxRestrictionP_ * exp(|ieta|*log(2.5)/18))
  // with the factor for exponential slopeRestrictionP_ = log(2.5)/18
  // maxRestrictionP_ = 8 GeV as came from a study
  std::string labelBS = iConfig.getParameter<std::string>("labelBeamSpot");
  edm::InputTag algTag = iConfig.getParameter<edm::InputTag>("algInputTag");
  edm::InputTag extTag = iConfig.getParameter<edm::InputTag>("extInputTag");
  std::string labelMuon = iConfig.getParameter<std::string>("labelMuon");

  for (unsigned int k = 0; k < oldID_.size(); ++k) {
    oldDet_.emplace_back((oldID_[k] / 10000) % 10);
    oldEta_.emplace_back((oldID_[k] / 100) % 100);
    oldDepth_.emplace_back(oldID_[k] % 100);
  }
  l1GtUtils_ = new l1t::L1TGlobalUtil(iConfig, consumesCollector(), *this, algTag, extTag, l1t::UseEventSetupIn::Event);
  // define tokens for access
  tok_trigEvt_ = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_bs_ = consumes<reco::BeamSpot>(labelBS);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  tok_parts_ = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  tok_cala_ = consumes<CaloTowerCollection>(labelTower_);
  tok_alg_ = consumes<BXVector<GlobalAlgBlk>>(algTag);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon);
  tok_recVtx_ = consumes<reco::VertexCollection>(labelRecVtx_);
  tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEB_));
  tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEE_));
  tok_hbhe_ = consumes<HBHERecHitCollection>(labelHBHE_);
  edm::LogVerbatim("HcalIsoTrack") << "Labels used " << triggerEvent_ << " " << theTriggerResultsLabel_ << " "
                                   << labelBS << " " << labelRecVtx_ << " " << labelGenTrack_ << " "
                                   << edm::InputTag("ecalRecHit", labelEB_) << " "
                                   << edm::InputTag("ecalRecHit", labelEE_) << " " << labelHBHE_ << " " << labelTower_
                                   << " " << labelMuon;

  tok_ddrec_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  tok_bFieldH_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_ecalChStatus_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  tok_sevlv_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_resp_ = esConsumes<HcalRespCorrs, HcalRespCorrsRcd>();
  tok_ecalPFRecHitThresholds_ = esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>();

  edm::LogVerbatim("HcalIsoTrack")
      << "Parameters read from config file \n"
      << "\t minPt " << selectionParameter_.minPt << "\t theTrackQuality " << theTrackQuality_ << "\t minQuality "
      << selectionParameter_.minQuality << "\t maxDxyPV " << selectionParameter_.maxDxyPV << "\t maxDzPV "
      << selectionParameter_.maxDzPV << "\t maxChi2 " << selectionParameter_.maxChi2 << "\t maxDpOverP "
      << selectionParameter_.maxDpOverP << "\t minOuterHit " << selectionParameter_.minOuterHit << "\t minLayerCrossed "
      << selectionParameter_.minLayerCrossed << "\t maxInMiss " << selectionParameter_.maxInMiss << "\t maxOutMiss "
      << selectionParameter_.maxOutMiss << "\t a_coneR " << a_coneR_ << ":" << a_coneR1_ << ":" << a_coneR2_
      << "\t a_charIsoR " << a_charIsoR_ << "\t a_mipR " << a_mipR_ << "\t a_mipR2 " << a_mipR2_ << "\t a_mipR3 "
      << a_mipR3_ << "\t a_mipR4 " << a_mipR4_ << "\t a_mipR5 " << a_mipR5_ << "\n pTrackMin_ " << pTrackMin_
      << "\t eEcalMax_ " << eEcalMax_ << "\t maxRestrictionP_ " << maxRestrictionP_ << "\t slopeRestrictionP_ "
      << slopeRestrictionP_ << "\t eIsolateStrong_ " << eIsolate1_ << "\t eIsolateSoft_ " << eIsolate2_
      << "\t hcalScale_ " << hcalScale_ << "\n\t momentumLow_ " << pTrackLow_ << "\t momentumHigh_ " << pTrackHigh_
      << "\n\t ignoreTrigger_ " << ignoreTrigger_ << "\n\t useL1Trigger_ " << useL1Trigger_ << "\t unCorrect_     "
      << unCorrect_ << "\t collapseDepth_ " << collapseDepth_ << "\t L1TrigName_    " << l1TrigName_
      << "\nThreshold flag used " << usePFThresh_ << " value for EB " << hitEthrEB_ << " EE " << hitEthrEE0_ << ":"
      << hitEthrEE1_ << ":" << hitEthrEE2_ << ":" << hitEthrEE3_ << ":" << hitEthrEELo_ << ":" << hitEthrEEHi_;
  edm::LogVerbatim("HcalIsoTrack") << "Process " << processName_ << " L1Filter:" << l1Filter_
                                   << " L2Filter:" << l2Filter_ << " L3Filter:" << l3Filter_ << " and "
                                   << debEvents_.size() << " events to be debugged";
  for (unsigned int k = 0; k < trigNames_.size(); ++k) {
    edm::LogVerbatim("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];
  }
  edm::LogVerbatim("HcalIsoTrack") << oldID_.size() << " DetIDs to be corrected with HEP17 flag:" << hep17_;
  for (unsigned int k = 0; k < oldID_.size(); ++k)
    edm::LogVerbatim("HcalIsoTrack") << "[" << k << "] Det " << oldDet_[k] << " EtaAbs " << oldEta_[k] << " Depth "
                                     << oldDepth_[k] << ":" << newDepth_[k];

  for (int i = 0; i < 10; i++)
    phibins_.push_back(-M_PI + 0.1 * (2 * i + 1) * M_PI);
  for (int i = 0; i < 8; ++i)
    etabins_.push_back(-2.1 + 0.6 * i);
  etadist_ = etabins_[1] - etabins_[0];
  phidist_ = phibins_[1] - phibins_[0];
  etahalfdist_ = 0.5 * etadist_;
  phihalfdist_ = 0.5 * phidist_;
  edm::LogVerbatim("HcalIsoTrack") << "EtaDist " << etadist_ << " " << etahalfdist_ << " PhiDist " << phidist_ << " "
                                   << phihalfdist_;
  //create also IsolatedPixelTrackCandidateCollection which contains isolation info and reference to primary track
  produces<HcalIsoTrkCalibVariablesCollection>(labelIsoTkVar_);
  produces<HcalIsoTrkEventVariablesCollection>(labelIsoTkEvtVar_);
  edm::LogVerbatim("HcalIsoTrack") << " Expected to produce the collections:\n"
                                   << "HcalIsoTrkCalibVariablesCollection with label " << labelIsoTkVar_
                                   << "\nand HcalIsoTrkEventVariablesCollection with label " << labelIsoTkEvtVar_;
}

void AlCaHcalIsotrkProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "HLT");
  std::vector<std::string> trig;
  desc.add<std::vector<std::string>>("triggers", trig);
  desc.add<std::string>("l1Filter", "");
  desc.add<std::string>("l2Filter", "L2Filter");
  desc.add<std::string>("l3Filter", "Filter");
  // following 10 parameters are parameters to select good tracks
  desc.add<std::string>("trackQuality", "highPurity");
  desc.add<double>("minTrackPt", 1.0);
  desc.add<double>("maxDxyPV", 0.02);
  desc.add<double>("maxDzPV", 0.02);
  desc.add<double>("maxChi2", 5.0);
  desc.add<double>("maxDpOverP", 0.1);
  desc.add<int>("minOuterHit", 4);
  desc.add<int>("minLayerCrossed", 8);
  desc.add<int>("maxInMiss", 0);
  desc.add<int>("maxOutMiss", 0);
  // Minimum momentum of selected isolated track and signal zone
  desc.add<double>("minimumTrackP", 10.0);
  desc.add<double>("coneRadius", 34.98);
  // signal zone in ECAL and MIP energy cutoff
  desc.add<double>("coneRadiusMIP", 14.0);
  desc.add<double>("coneRadiusMIP2", 18.0);
  desc.add<double>("coneRadiusMIP3", 20.0);
  desc.add<double>("coneRadiusMIP4", 22.0);
  desc.add<double>("coneRadiusMIP5", 24.0);
  desc.add<double>("maximumEcalEnergy", 2.0);
  // following 4 parameters are for isolation cuts and described in the code
  desc.add<double>("maxTrackP", 8.0);
  desc.add<double>("slopeTrackP", 0.05090504066);
  desc.add<double>("isolationEnergyTight", 2.0);
  desc.add<double>("isolationEnergyLoose", 10.0);
  // energy thershold for ECAL (from Egamma group)
  desc.add<double>("EBHitEnergyThreshold", 0.08);
  desc.add<double>("EEHitEnergyThreshold0", 0.30);
  desc.add<double>("EEHitEnergyThreshold1", 0.00);
  desc.add<double>("EEHitEnergyThreshold2", 0.00);
  desc.add<double>("EEHitEnergyThreshold3", 0.00);
  desc.add<double>("EEHitEnergyThresholdLow", 0.30);
  desc.add<double>("EEHitEnergyThresholdHigh", 0.30);
  // prescale factors
  desc.add<double>("momentumLow", 40.0);
  desc.add<double>("momentumHigh", 60.0);
  desc.add<int>("prescaleLow", 1);
  desc.add<int>("prescaleHigh", 1);
  // various labels for collections used in the code
  desc.add<edm::InputTag>("labelTriggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<edm::InputTag>("labelTriggerResult", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<std::string>("labelTrack", "generalTracks");
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelEBRecHit", "EcalRecHitsEB");
  desc.add<std::string>("labelEERecHit", "EcalRecHitsEE");
  desc.add<std::string>("labelHBHERecHit", "hbhereco");
  desc.add<std::string>("labelBeamSpot", "offlineBeamSpot");
  desc.add<std::string>("labelCaloTower", "towerMaker");
  desc.add<std::string>("labelMuon", "muons");
  desc.add<edm::InputTag>("algInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("extInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<std::string>("isoTrackLabel", "HcalIsoTrack");
  desc.add<std::string>("isoTrackEventLabel", "HcalIsoTrackEvent");
  //  Various flags used for selecting tracks, choice of energy Method2/0
  //  Data type 0/1 for single jet trigger or others
  desc.addUntracked<bool>("ignoreTriggers", false);
  desc.addUntracked<bool>("useL1Trigger", false);
  desc.addUntracked<double>("hcalScale", 1.0);
  desc.addUntracked<bool>("unCorrect", false);
  desc.addUntracked<bool>("collapseDepth", false);
  desc.addUntracked<std::string>("l1TrigName", "L1_SingleJet60");
  std::vector<int> dummy;
  desc.addUntracked<std::vector<int>>("oldID", dummy);
  desc.addUntracked<std::vector<int>>("newDepth", dummy);
  desc.addUntracked<bool>("hep17", false);
  std::vector<int> events;
  desc.add<std::vector<int>>("debugEvents", events);
  desc.add<bool>("usePFThreshold", true);
  descriptions.add("alcaHcalIsotrkProducer", desc);
}

void AlCaHcalIsotrkProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  nAll_++;
  debug_ = (debEvents_.empty())
               ? true
               : (std::find(debEvents_.begin(), debEvents_.end(), iEvent.id().event()) != debEvents_.end());
#ifdef EDM_ML_DEBUG
  if (debug_)
    edm::LogVerbatim("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                     << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                     << iEvent.bunchCrossing();
#endif

  HcalIsoTrkEventVariables isoTrkEvent;
  //Step1: Get all the relevant containers
  const MagneticField* bField = &iSetup.getData(tok_bFieldH_);
  const EcalChannelStatus* theEcalChStatus = &iSetup.getData(tok_ecalChStatus_);
  const EcalSeverityLevelAlgo* theEcalSevlv = &iSetup.getData(tok_sevlv_);
  eThresholds_ = &iSetup.getData(tok_ecalPFRecHitThresholds_);

  // get calogeometry and calotopology
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const CaloTopology* caloTopology = &iSetup.getData(tok_caloTopology_);
  const HcalTopology* theHBHETopology = &iSetup.getData(tok_htopo_);

  // get Hcal response corrections
  const HcalRespCorrs* respCorrs = &iSetup.getData(tok_resp_);

  bool okC(true);
  //Get track collection
  auto trkCollection = iEvent.getHandle(tok_genTrack_);
  if (!trkCollection.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    okC = false;
  }

  //Get muon collection
  auto const& muonh = iEvent.getHandle(tok_Muon_);

  //Define the best vertex and the beamspot
  auto const& recVtxs = iEvent.getHandle(tok_recVtx_);
  auto const& beamSpotH = iEvent.getHandle(tok_bs_);
  math::XYZPoint leadPV(0, 0, 0);
  int goodPV(0);
  if (recVtxs.isValid() && !(recVtxs->empty())) {
    isoTrkEvent.allvertex_ = recVtxs->size();
    for (unsigned int k = 0; k < recVtxs->size(); ++k) {
      if (!((*recVtxs)[k].isFake()) && ((*recVtxs)[k].ndof() > 4)) {
        if (goodPV == 0)
          leadPV = math::XYZPoint((*recVtxs)[k].x(), (*recVtxs)[k].y(), (*recVtxs)[k].z());
        goodPV++;
      }
    }
  }
  if (goodPV == 0 && beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
#ifdef EDM_ML_DEBUG
  if (debug_) {
    edm::LogVerbatim("HcalIsoTrack") << "Primary Vertex " << leadPV << " out of " << goodPV << " vertex";
    if (beamSpotH.isValid())
      edm::LogVerbatim("HcalIsoTrack") << " Beam Spot " << beamSpotH->position();
  }
#endif

  // RecHits
  auto barrelRecHitsHandle = iEvent.getHandle(tok_EB_);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
    okC = false;
  }
  auto endcapRecHitsHandle = iEvent.getHandle(tok_EE_);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
    okC = false;
  }
  auto hbhe = iEvent.getHandle(tok_hbhe_);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
    okC = false;
  }
  auto caloTower = iEvent.getHandle(tok_cala_);

  //=== genParticle information
  auto const& genParticles = iEvent.getHandle(tok_parts_);
  auto const& genEventInfo = iEvent.getHandle(tok_ew_);
  double eventWeight = (genEventInfo.isValid()) ? genEventInfo->weight() : 1.0;

  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, false);
  std::vector<spr::propagatedTrackID> trkCaloDets;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDets, false);
  std::vector<math::XYZTLorentzVector> vecL1, vecL3;
  isoTrkEvent.tracks_ = trkCollection->size();
  isoTrkEvent.tracksProp_ = trkCaloDirections.size();
  isoTrkEvent.hltbits_.assign(trigNames_.size(), false);

  if (!ignoreTrigger_) {
    //L1
    l1GtUtils_->retrieveL1(iEvent, iSetup, tok_alg_);
    const std::vector<std::pair<std::string, bool>>& finalDecisions = l1GtUtils_->decisionsFinal();
    for (const auto& decision : finalDecisions) {
      if (decision.first.find(l1TrigName_) != std::string::npos) {
        isoTrkEvent.l1Bit_ = decision.second;
        break;
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug_)
      edm::LogVerbatim("HcalIsoTrack") << "Trigger Information for " << l1TrigName_ << " is " << isoTrkEvent.l1Bit_
                                       << " from a list of " << finalDecisions.size() << " decisions";
#endif

    //HLT
    auto const& triggerResults = iEvent.getHandle(tok_trigRes_);
    if (triggerResults.isValid()) {
      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string>& names = triggerNames.triggerNames();
      if (!trigNames_.empty()) {
        for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          int hlt = triggerResults->accept(iHLT);
          for (unsigned int i = 0; i < trigNames_.size(); ++i) {
            if (names[iHLT].find(trigNames_[i]) != std::string::npos) {
              isoTrkEvent.hltbits_[i] = (hlt > 0);
              if (hlt > 0)
                isoTrkEvent.trigPass_ = true;
#ifdef EDM_ML_DEBUG
              if (debug_)
                edm::LogVerbatim("HcalIsoTrack")
                    << "This trigger " << names[iHLT] << " Flag " << hlt << ":" << isoTrkEvent.hltbits_[i];
#endif
            }
          }
        }
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug_)
      edm::LogVerbatim("HcalIsoTrack") << "HLT Information shows " << isoTrkEvent.trigPass_ << ":" << trigNames_.empty()
                                       << ":" << okC;
#endif
  }

  auto outputHcalIsoTrackColl = std::make_unique<HcalIsoTrkCalibVariablesCollection>();
  std::array<int, 3> ntksave{{0, 0, 0}};
  if (ignoreTrigger_ || useL1Trigger_) {
    if (ignoreTrigger_ || isoTrkEvent.l1Bit_)
      ntksave = getProducts(goodPV,
                            eventWeight,
                            vecL1,
                            vecL3,
                            leadPV,
                            trkCaloDirections,
                            trkCaloDets,
                            geo,
                            caloTopology,
                            theHBHETopology,
                            theEcalChStatus,
                            theEcalSevlv,
                            barrelRecHitsHandle,
                            endcapRecHitsHandle,
                            hbhe,
                            caloTower,
                            genParticles,
                            respCorrs,
                            muonh,
                            *outputHcalIsoTrackColl,
                            isoTrkEvent,
                            iEvent.id());
    isoTrkEvent.tracksSaved_ = ntksave[0];
    isoTrkEvent.tracksLoose_ = ntksave[1];
    isoTrkEvent.tracksTight_ = ntksave[2];
  } else {
    trigger::TriggerEvent triggerEvent;
    auto const& triggerEventHandle = iEvent.getHandle(tok_trigEvt_);
    if (!triggerEventHandle.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product " << triggerEvent_.label();
    } else if (okC) {
      triggerEvent = *(triggerEventHandle.product());
      const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
      bool done(false);
      auto const& triggerResults = iEvent.getHandle(tok_trigRes_);
      if (triggerResults.isValid()) {
        std::vector<std::string> modules;
        const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
        const std::vector<std::string>& names = triggerNames.triggerNames();
        for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          bool ok = (isoTrkEvent.trigPass_) || (trigNames_.empty());
          if (ok) {
            unsigned int triggerindx = hltConfig_.triggerIndex(names[iHLT]);
            const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
            std::vector<math::XYZTLorentzVector> vecL2;
            vecL1.clear();
            vecL3.clear();
            //loop over all trigger filters in event (i.e. filters passed)
            for (unsigned int ifilter = 0; ifilter < triggerEvent.sizeFilters(); ++ifilter) {
              std::vector<int> Keys;
              auto const label = triggerEvent.filterLabel(ifilter);
              //loop over keys to objects passing this filter
              for (unsigned int imodule = 0; imodule < moduleLabels.size(); imodule++) {
                if (label.find(moduleLabels[imodule]) != label.npos) {
#ifdef EDM_ML_DEBUG
                  if (debug_)
                    edm::LogVerbatim("HcalIsoTrack") << "FilterName " << label;
#endif
                  for (unsigned int ifiltrKey = 0; ifiltrKey < triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
                    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
                    const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
                    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
                    if (label.find(l2Filter_) != std::string::npos) {
                      vecL2.push_back(v4);
                    } else if (label.find(l3Filter_) != std::string::npos) {
                      vecL3.push_back(v4);
                    } else if ((label.find(l1Filter_) != std::string::npos) || (l1Filter_.empty())) {
                      vecL1.push_back(v4);
                    }
#ifdef EDM_ML_DEBUG
                    if (debug_)
                      edm::LogVerbatim("HcalIsoTrack")
                          << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta() << " phi " << TO.phi()
                          << " mass " << TO.mass() << " Id " << TO.id();
#endif
                  }
#ifdef EDM_ML_DEBUG
                  if (debug_)
                    edm::LogVerbatim("HcalIsoTrack")
                        << "sizes " << vecL1.size() << ":" << vecL2.size() << ":" << vecL3.size();
#endif
                }
              }
            }

            // Now Make the products for all selected tracks
            if (!done) {
              ntksave = getProducts(goodPV,
                                    eventWeight,
                                    vecL1,
                                    vecL3,
                                    leadPV,
                                    trkCaloDirections,
                                    trkCaloDets,
                                    geo,
                                    caloTopology,
                                    theHBHETopology,
                                    theEcalChStatus,
                                    theEcalSevlv,
                                    barrelRecHitsHandle,
                                    endcapRecHitsHandle,
                                    hbhe,
                                    caloTower,
                                    genParticles,
                                    respCorrs,
                                    muonh,
                                    *outputHcalIsoTrackColl,
                                    isoTrkEvent,
                                    iEvent.id());
              isoTrkEvent.tracksSaved_ += ntksave[0];
              isoTrkEvent.tracksLoose_ += ntksave[1];
              isoTrkEvent.tracksTight_ += ntksave[2];
              done = true;
            }
          }
        }
      }
    }
  }
  isoTrkEvent.trigPassSel_ = (isoTrkEvent.tracksSaved_ > 0);
  if (isoTrkEvent.trigPassSel_) {
    ++nGood_;
    for (auto itr = outputHcalIsoTrackColl->begin(); itr != outputHcalIsoTrackColl->end(); ++itr) {
      if (itr->p_ > pTrackLow_ && itr->p_ < pTrackHigh_)
        ++nRange_;
    }
  }

  auto outputEventcol = std::make_unique<HcalIsoTrkEventVariablesCollection>();
  outputEventcol->emplace_back(isoTrkEvent);
  iEvent.put(std::move(outputEventcol), labelIsoTkEvtVar_);
  iEvent.put(std::move(outputHcalIsoTrackColl), labelIsoTkVar_);
}

void AlCaHcalIsotrkProducer::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
  globalCache()->nRange_ += nRange_;
}

void AlCaHcalIsotrkProducer::globalEndJob(const alCaHcalIsotrkProducer::Counters* count) {
  edm::LogVerbatim("HcalIsoTrack") << "Finds " << count->nGood_ << " good tracks in " << count->nAll_ << " events and "
                                   << count->nRange_ << " events in the momentum range";
}

void AlCaHcalIsotrkProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  hdc_ = &iSetup.getData(tok_ddrec_);
  if (!ignoreTrigger_) {
    bool changed(false);
    edm::LogVerbatim("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                                     << hltConfig_.init(iRun, iSetup, processName_, changed);
  }
}

void AlCaHcalIsotrkProducer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogVerbatim("HcalIsoTrack") << "endRun [" << nRun_ << "] " << iRun.run();
  ++nRun_;
}

std::array<int, 3> AlCaHcalIsotrkProducer::getProducts(int goodPV,
                                                       double eventWeight,
                                                       std::vector<math::XYZTLorentzVector>& vecL1,
                                                       std::vector<math::XYZTLorentzVector>& vecL3,
                                                       math::XYZPoint& leadPV,
                                                       std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
                                                       std::vector<spr::propagatedTrackID>& trkCaloDets,
                                                       const CaloGeometry* geo,
                                                       const CaloTopology* caloTopology,
                                                       const HcalTopology* theHBHETopology,
                                                       const EcalChannelStatus* theEcalChStatus,
                                                       const EcalSeverityLevelAlgo* theEcalSevlv,
                                                       edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
                                                       edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
                                                       edm::Handle<HBHERecHitCollection>& hbhe,
                                                       edm::Handle<CaloTowerCollection>& tower,
                                                       const edm::Handle<reco::GenParticleCollection>& genParticles,
                                                       const HcalRespCorrs* respCorrs,
                                                       const edm::Handle<reco::MuonCollection>& muonh,
                                                       std::vector<HcalIsoTrkCalibVariables>& hocalib,
                                                       HcalIsoTrkEventVariables& hocalibEvent,
                                                       const edm::EventID& eventId) {
  int nSave(0), nLoose(0), nTight(0);
  unsigned int nTracks(0);
  double rhohEV = (tower.isValid()) ? rhoh(tower) : 0;

  //Loop over tracks
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
       trkDetItr++, nTracks++) {
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
    bool accept(false);
    HcalIsoTrkCalibVariables isoTk;
    isoTk.eventWeight_ = eventWeight;
    isoTk.goodPV_ = goodPV;
    isoTk.nVtx_ = hocalibEvent.allvertex_;
    isoTk.nTrk_ = trkCaloDirections.size();
    isoTk.rhoh_ = rhohEV;
    for (const auto& trig : hocalibEvent.hltbits_)
      isoTk.trgbits_.emplace_back(trig);
    if (!vecL1.empty()) {
      isoTk.l1pt_ = vecL1[0].pt();
      isoTk.l1eta_ = vecL1[0].eta();
      isoTk.l1phi_ = vecL1[0].phi();
    }
    if (!vecL3.empty()) {
      isoTk.l3pt_ = vecL3[0].pt();
      isoTk.l3eta_ = vecL3[0].eta();
      isoTk.l3phi_ = vecL3[0].phi();
    }

    isoTk.p_ = pTrack->p();
    isoTk.pt_ = pTrack->pt();
    isoTk.phi_ = pTrack->phi();
#ifdef EDM_ML_DEBUG
    if (debug_)
      edm::LogVerbatim("HcalIsoTrack") << "This track : " << nTracks << " (pt|eta|phi|p) : " << isoTk.pt_ << "|"
                                       << pTrack->eta() << "|" << isoTk.phi_ << "|" << isoTk.p_;
    int flag(0);
#endif
    isoTk.mindR2_ = 999;
    for (unsigned int k = 0; k < vecL3.size(); ++k) {
      double dr = dR(vecL3[k], v4);
      if (dr < isoTk.mindR2_) {
        isoTk.mindR2_ = dr;
      }
    }
    isoTk.mindR1_ = (!vecL1.empty()) ? dR(vecL1[0], v4) : 999;
#ifdef EDM_ML_DEBUG
    if (debug_)
      edm::LogVerbatim("HcalIsoTrack") << "Closest L3 object at dr : " << isoTk.mindR2_ << " and from L1 "
                                       << isoTk.mindR1_;
#endif
    isoTk.ieta_ = isoTk.iphi_ = 0;
    if (trkDetItr->okHCAL) {
      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
      isoTk.ieta_ = detId.ieta();
      isoTk.iphi_ = detId.iphi();
      if (isoTk.p_ > 40.0 && isoTk.p_ <= 60.0)
        hocalibEvent.ietaAll_.emplace_back(isoTk.ieta_);
    }
    //Selection of good track
    isoTk.selectTk_ = spr::goodTrack(pTrack, leadPV, selectionParameter_, false);
    spr::trackSelectionParameters oneCutParameters = selectionParameter_;
    oneCutParameters.maxDxyPV = 10;
    oneCutParameters.maxDzPV = 100;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss = 2;
    bool qltyFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    oneCutParameters = selectionParameter_;
    oneCutParameters.maxDxyPV = 10;
    oneCutParameters.maxDzPV = 100;
    isoTk.qltyMissFlag_ = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    oneCutParameters = selectionParameter_;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss = 2;
    isoTk.qltyPVFlag_ = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    double eIsolation = maxRestrictionP_ * exp(slopeRestrictionP_ * std::abs((double)isoTk.ieta_));
    if (eIsolation < eIsolate1_)
      eIsolation = eIsolate1_;
    if (eIsolation < eIsolate2_)
      eIsolation = eIsolate2_;
#ifdef EDM_ML_DEBUG
    if (debug_)
      edm::LogVerbatim("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" << trkDetItr->okECAL << "|"
                                       << trkDetItr->okHCAL << " eIsolation " << eIsolation;
    if (qltyFlag)
      flag += 1;
    if (trkDetItr->okECAL)
      flag += 2;
    if (trkDetItr->okHCAL)
      flag += 4;
#endif
    isoTk.qltyFlag_ = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
    bool notMuon = (muonh.isValid()) ? notaMuon(pTrack, muonh) : true;
#ifdef EDM_ML_DEBUG
    if (notMuon)
      flag += 8;
#endif
    if (isoTk.qltyFlag_ && notMuon) {
      int nNearTRKs(0);
      ////////////////////////////////-MIP STUFF-//////////////////////////////
      std::vector<DetId> eIds;
      std::vector<double> eHit;
#ifdef EDM_ML_DEBUG
      double eMipDR =
#endif
          spr::eCone_ecal(geo,
                          barrelRecHitsHandle,
                          endcapRecHitsHandle,
                          trkDetItr->pointHCAL,
                          trkDetItr->pointECAL,
                          a_mipR_,
                          trkDetItr->directionECAL,
                          eIds,
                          eHit);
      double eEcal(0);
      for (unsigned int k = 0; k < eIds.size(); ++k) {
        if (eHit[k] > eThreshold(eIds[k], geo))
          eEcal += eHit[k];
      }
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR << ":" << eEcal;
#endif
      isoTk.eMipDR_.emplace_back(eEcal);
      ////////////////////////////////-MIP STUFF-///////////////////////////////
      ////////////////////////////////-MIP STUFF-2//////////////////////////////
      std::vector<DetId> eIds2;
      std::vector<double> eHit2;
#ifdef EDM_ML_DEBUG
      double eMipDR2 =
#endif
          spr::eCone_ecal(geo,
                          barrelRecHitsHandle,
                          endcapRecHitsHandle,
                          trkDetItr->pointHCAL,
                          trkDetItr->pointECAL,
                          a_mipR2_,
                          trkDetItr->directionECAL,
                          eIds2,
                          eHit2);
      double eEcal2(0);
      for (unsigned int k = 0; k < eIds2.size(); ++k) {
        if (eHit2[k] > eThreshold(eIds2[k], geo))
          eEcal2 += eHit2[k];
      }
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR2 << ":" << eEcal2;
#endif
      isoTk.eMipDR_.emplace_back(eEcal2);
      ////////////////////////////////-MIP STUFF-2/////////////////////////////
      ////////////////////////////////-MIP STUFF-3/////////////////////////////
      std::vector<DetId> eIds3;
      std::vector<double> eHit3;
#ifdef EDM_ML_DEBUG
      double eMipDR3 =
#endif
          spr::eCone_ecal(geo,
                          barrelRecHitsHandle,
                          endcapRecHitsHandle,
                          trkDetItr->pointHCAL,
                          trkDetItr->pointECAL,
                          a_mipR3_,
                          trkDetItr->directionECAL,
                          eIds3,
                          eHit3);
      double eEcal3(0);
      for (unsigned int k = 0; k < eIds3.size(); ++k) {
        if (eHit3[k] > eThreshold(eIds3[k], geo))
          eEcal3 += eHit3[k];
      }
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR3 << ":" << eEcal3;
#endif
      isoTk.eMipDR_.emplace_back(eEcal3);
      ////////////////////////////////-MIP STUFF-3/////////////////////////////
      ////////////////////////////////-MIP STUFF-4/////////////////////////////
      std::vector<DetId> eIds4;
      std::vector<double> eHit4;
#ifdef EDM_ML_DEBUG
      double eMipDR4 =
#endif
          spr::eCone_ecal(geo,
                          barrelRecHitsHandle,
                          endcapRecHitsHandle,
                          trkDetItr->pointHCAL,
                          trkDetItr->pointECAL,
                          a_mipR4_,
                          trkDetItr->directionECAL,
                          eIds4,
                          eHit4);
      double eEcal4(0);
      for (unsigned int k = 0; k < eIds4.size(); ++k) {
        if (eHit4[k] > eThreshold(eIds4[k], geo))
          eEcal4 += eHit4[k];
      }
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR4 << ":" << eEcal4;
#endif
      isoTk.eMipDR_.emplace_back(eEcal4);
      ////////////////////////////////-MIP STUFF-4/////////////////////////////
      ////////////////////////////////-MIP STUFF-5/////////////////////////////
      std::vector<DetId> eIds5;
      std::vector<double> eHit5;
#ifdef EDM_ML_DEBUG
      double eMipDR5 =
#endif
          spr::eCone_ecal(geo,
                          barrelRecHitsHandle,
                          endcapRecHitsHandle,
                          trkDetItr->pointHCAL,
                          trkDetItr->pointECAL,
                          a_mipR5_,
                          trkDetItr->directionECAL,
                          eIds5,
                          eHit5);
      double eEcal5(0);
      for (unsigned int k = 0; k < eIds5.size(); ++k) {
        if (eHit5[k] > eThreshold(eIds5[k], geo))
          eEcal5 += eHit5[k];
      }
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << eMipDR5 << ":" << eEcal5;
#endif
      isoTk.eMipDR_.emplace_back(eEcal5);
      ////////////////////////////////-MIP STUFF-5/////////////////////////////

      isoTk.emaxNearP_ = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 15, 15);
      const DetId cellE(trkDetItr->detIdECAL);
      std::pair<double, bool> e11x11P = spr::eECALmatrix(cellE,
                                                         barrelRecHitsHandle,
                                                         endcapRecHitsHandle,
                                                         *theEcalChStatus,
                                                         geo,
                                                         caloTopology,
                                                         theEcalSevlv,
                                                         5,
                                                         5,
                                                         -100.0,
                                                         -100.0,
                                                         -100.0,
                                                         100.0);
      std::pair<double, bool> e15x15P = spr::eECALmatrix(cellE,
                                                         barrelRecHitsHandle,
                                                         endcapRecHitsHandle,
                                                         *theEcalChStatus,
                                                         geo,
                                                         caloTopology,
                                                         theEcalSevlv,
                                                         7,
                                                         7,
                                                         -100.0,
                                                         -100.0,
                                                         -100.0,
                                                         100.0);
      if (e11x11P.second && e15x15P.second) {
        isoTk.eAnnular_ = (e15x15P.first - e11x11P.first);
      } else {
        isoTk.eAnnular_ = -(e15x15P.first - e11x11P.first);
      }
      isoTk.hmaxNearP_ = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, false);
      const DetId cellH(trkDetItr->detIdHCAL);
      double h5x5 = spr::eHCALmatrix(
          theHBHETopology, cellH, hbhe, 2, 2, false, true, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0);
      double h7x7 = spr::eHCALmatrix(
          theHBHETopology, cellH, hbhe, 3, 3, false, true, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0);
      isoTk.hAnnular_ = h7x7 - h5x5;
#ifdef EDM_ML_DEBUG
      if (debug_)
        edm::LogVerbatim("HcalIsoTrack") << "max p Near (Ecal) " << isoTk.emaxNearP_ << " (Hcal) " << isoTk.hmaxNearP_
                                         << " Annular E (Ecal) " << e11x11P.first << ":" << e15x15P.first << ":"
                                         << isoTk.eAnnular_ << " (Hcal) " << h5x5 << ":" << h7x7 << ":"
                                         << isoTk.hAnnular_;
      if (isoTk.eMipDR_[0] < eEcalMax_)
        flag += 16;
      if (isoTk.hmaxNearP_ < eIsolation)
        flag += 32;
#endif
      isoTk.gentrackP_ = trackP(pTrack, genParticles);
      if (isoTk.eMipDR_[0] < eEcalMax_ && isoTk.hmaxNearP_ < eIsolation) {
        int nRecHits(-999), nRecHits1(-999), nRecHits3(-999);
        std::vector<DetId> ids, ids1, ids3;
        std::vector<double> edet0, edet1, edet3;
        isoTk.eHcal_ = spr::eCone_hcal(geo,
                                       hbhe,
                                       trkDetItr->pointHCAL,
                                       trkDetItr->pointECAL,
                                       a_coneR_,
                                       trkDetItr->directionHCAL,
                                       nRecHits,
                                       ids,
                                       edet0,
                                       0);
        if (!oldID_.empty()) {
          for (unsigned k = 0; k < ids.size(); ++k)
            ids[k] = newId(ids[k]);
        }
        storeEnergy(respCorrs, ids, edet0, isoTk.eHcal_, isoTk.detIds_, isoTk.hitEnergies_);
        std::pair<double, double> ehcal0 =
            storeEnergy(respCorrs, hbhe, ids, isoTk.hitEnergiesRaw_, isoTk.hitEnergiesAux_);
        isoTk.eHcalRaw_ = ehcal0.first;
        isoTk.eHcalAux_ = ehcal0.second;

        //----- hcal energy in the extended cone 1 (a_coneR+10) --------------
        isoTk.eHcal10_ = spr::eCone_hcal(geo,
                                         hbhe,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_coneR1_,
                                         trkDetItr->directionHCAL,
                                         nRecHits1,
                                         ids1,
                                         edet1,
                                         0);
        if (!oldID_.empty()) {
          for (unsigned k = 0; k < ids1.size(); ++k)
            ids1[k] = newId(ids1[k]);
        }
        storeEnergy(respCorrs, ids1, edet1, isoTk.eHcal10_, isoTk.detIds1_, isoTk.hitEnergies1_);
        std::pair<double, double> ehcal1 =
            storeEnergy(respCorrs, hbhe, ids1, isoTk.hitEnergies1Raw_, isoTk.hitEnergies1Aux_);
        isoTk.eHcal10Raw_ = ehcal1.first;
        isoTk.eHcal10Aux_ = ehcal1.second;

        //----- hcal energy in the extended cone 3 (a_coneR+30) --------------
        isoTk.eHcal30_ = spr::eCone_hcal(geo,
                                         hbhe,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_coneR2_,
                                         trkDetItr->directionHCAL,
                                         nRecHits3,
                                         ids3,
                                         edet3,
                                         0);
        if (!oldID_.empty()) {
          for (unsigned k = 0; k < ids3.size(); ++k)
            ids3[k] = newId(ids3[k]);
        }
        storeEnergy(respCorrs, ids3, edet3, isoTk.eHcal30_, isoTk.detIds3_, isoTk.hitEnergies3_);
        std::pair<double, double> ehcal3 =
            storeEnergy(respCorrs, hbhe, ids3, isoTk.hitEnergies3Raw_, isoTk.hitEnergies3Aux_);
        isoTk.eHcal30Raw_ = ehcal3.first;
        isoTk.eHcal30Aux_ = ehcal3.second;

        if (isoTk.p_ > pTrackMin_)
          accept = true;
#ifdef EDM_ML_DEBUG
        if (accept)
          flag += 64;
        if (debug_) {
          std::string ctype = accept ? " ***** ACCEPT *****" : "";
          edm::LogVerbatim("HcalIsoTrack")
              << "This track : " << nTracks << " (pt|eta|phi|p) : " << isoTk.pt_ << "|" << pTrack->eta() << "|"
              << isoTk.phi_ << "|" << isoTk.p_ << " Generator Level p " << isoTk.gentrackP_;
          edm::LogVerbatim("HcalIsoTrack")
              << "e_MIP " << isoTk.eMipDR_[0] << " Chg Isolation " << isoTk.hmaxNearP_ << " eHcal" << isoTk.eHcal_
              << ":" << isoTk.eHcalRaw_ << ":" << isoTk.eHcalAux_ << " ieta " << isoTk.ieta_ << " Quality "
              << isoTk.qltyMissFlag_ << ":" << isoTk.qltyPVFlag_ << ":" << isoTk.selectTk_ << ctype;
          for (unsigned int ll = 0; ll < isoTk.detIds_.size(); ll++) {
            edm::LogVerbatim("HcalIsoTrack")
                << "det id is = " << HcalDetId(isoTk.detIds_[ll]) << "   hit enery is  = " << isoTk.hitEnergies_[ll]
                << " : " << isoTk.hitEnergiesRaw_[ll] << " : " << isoTk.hitEnergiesAux_[ll];
          }
          for (unsigned int ll = 0; ll < isoTk.detIds1_.size(); ll++) {
            edm::LogVerbatim("HcalIsoTrack")
                << "det id is = " << HcalDetId(isoTk.detIds1_[ll]) << "   hit enery is  = " << isoTk.hitEnergies1_[ll]
                << " : " << isoTk.hitEnergies1Raw_[ll] << " : " << isoTk.hitEnergies1Aux_[ll];
          }
          for (unsigned int ll = 0; ll < isoTk.detIds3_.size(); ll++) {
            edm::LogVerbatim("HcalIsoTrack")
                << "det id is = " << HcalDetId(isoTk.detIds3_[ll]) << "   hit enery is  = " << isoTk.hitEnergies3_[ll]
                << " : " << isoTk.hitEnergies3Raw_[ll] << " : " << isoTk.hitEnergies3Aux_[ll];
          }
        }
#endif
        if (accept) {
          edm::LogVerbatim("HcalIsoTrackX")
              << "Run " << eventId.run() << " Event " << eventId.event() << " Track " << nTracks << " p " << isoTk.p_;
          hocalib.emplace_back(isoTk);
          nSave++;
          int type(0);
          if (isoTk.eMipDR_[0] < 1.0) {
            if (isoTk.hmaxNearP_ < eIsolate2_) {
              ++nLoose;
              type = 1;
            }
            if (isoTk.hmaxNearP_ < eIsolate1_) {
              ++nTight;
              type = 2;
            }
          }
          if (isoTk.p_ > 40.0 && isoTk.p_ <= 60.0 && isoTk.selectTk_) {
            hocalibEvent.ietaGood_.emplace_back(isoTk.ieta_);
            hocalibEvent.trackType_.emplace_back(type);
          }
#ifdef EDM_ML_DEBUG
          for (unsigned int k = 0; k < isoTk.trgbits_.size(); k++) {
            edm::LogVerbatim("HcalIsoTrack") << "trigger bit is  = " << isoTk.trgbits_[k];
          }
#endif
        }
      }
    }
#ifdef EDM_ML_DEBUG
    if (debug_) {
      if (isoTk.eMipDR_.empty())
        edm::LogVerbatim("HcalIsoTrack") << "Track " << nTracks << " Selection Flag " << std::hex << flag << std::dec
                                         << " Accept " << accept << " Momentum " << isoTk.p_ << ":" << pTrackMin_;
      else
        edm::LogVerbatim("HcalIsoTrack") << "Track " << nTracks << " Selection Flag " << std::hex << flag << std::dec
                                         << " Accept " << accept << " Momentum " << isoTk.p_ << ":" << pTrackMin_
                                         << " Ecal Energy " << isoTk.eMipDR_[0] << ":" << eEcalMax_
                                         << " Charge Isolation " << isoTk.hmaxNearP_ << ":" << eIsolation;
    }
#endif
  }
  std::array<int, 3> i3{{nSave, nLoose, nTight}};
  return i3;
}

double AlCaHcalIsotrkProducer::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(), vec1.phi(), vec2.eta(), vec2.phi());
}

double AlCaHcalIsotrkProducer::trackP(const reco::Track* pTrack,
                                      const edm::Handle<reco::GenParticleCollection>& genParticles) {
  double pmom = -1.0;
  if (genParticles.isValid()) {
    double mindR(999.9);
    for (const auto& p : (*genParticles)) {
      double dR = reco::deltaR(pTrack->eta(), pTrack->phi(), p.momentum().Eta(), p.momentum().Phi());
      if (dR < mindR) {
        mindR = dR;
        pmom = p.momentum().R();
      }
    }
  }
  return pmom;
}

double AlCaHcalIsotrkProducer::rhoh(const edm::Handle<CaloTowerCollection>& tower) {
  std::vector<double> sumPFNallSMDQH2;
  sumPFNallSMDQH2.reserve(phibins_.size() * etabins_.size());
  for (auto eta : etabins_) {
    for (auto phi : phibins_) {
      double hadder = 0;
      for (const auto& pf_it : (*tower)) {
        if (fabs(eta - pf_it.eta()) > etahalfdist_)
          continue;
        if (fabs(reco::deltaPhi(phi, pf_it.phi())) > phihalfdist_)
          continue;
        hadder += pf_it.hadEt();
      }
      sumPFNallSMDQH2.emplace_back(hadder);
    }
  }

  double evt_smdq(0);
  std::sort(sumPFNallSMDQH2.begin(), sumPFNallSMDQH2.end());
  if (sumPFNallSMDQH2.size() % 2)
    evt_smdq = sumPFNallSMDQH2[(sumPFNallSMDQH2.size() - 1) / 2];
  else
    evt_smdq = (sumPFNallSMDQH2[sumPFNallSMDQH2.size() / 2] + sumPFNallSMDQH2[(sumPFNallSMDQH2.size() - 2) / 2]) / 2.;
  double rhoh = evt_smdq / (etadist_ * phidist_);
#ifdef EDM_ML_DEBUG
  if (debug_)
    edm::LogVerbatim("HcalIsoTrack") << "Rho " << evt_smdq << ":" << rhoh;
#endif
  return rhoh;
}

double AlCaHcalIsotrkProducer::eThreshold(const DetId& id, const CaloGeometry* geo) const {
  const GlobalPoint& pos = geo->getPosition(id);
  double eta = std::abs(pos.eta());
  double eThr(hitEthrEB_);
  if (usePFThresh_) {
    eThr = static_cast<double>((*eThresholds_)[id]);
  } else {
    if (id.subdetId() != EcalBarrel) {
      eThr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
      if (eThr < hitEthrEELo_)
        eThr = hitEthrEELo_;
      else if (eThr > hitEthrEEHi_)
        eThr = hitEthrEEHi_;
    }
  }
  return eThr;
}

DetId AlCaHcalIsotrkProducer::newId(const DetId& id) {
  HcalDetId hid(id);
  if (hep17_ && ((hid.iphi() < 63) || (hid.iphi() > 66) || (hid.zside() < 0)))
    return id;
  for (unsigned int k = 0; k < oldID_.size(); ++k) {
    if ((hid.subdetId() == oldDet_[k]) && (hid.ietaAbs() == oldEta_[k]) && (hid.depth() == oldDepth_[k])) {
      return static_cast<DetId>(HcalDetId(hid.subdet(), hid.ieta(), hid.iphi(), newDepth_[k]));
    }
  }
  return id;
}

void AlCaHcalIsotrkProducer::storeEnergy(const HcalRespCorrs* respCorrs,
                                         const std::vector<DetId>& ids,
                                         std::vector<double>& edet,
                                         double& eHcal,
                                         std::vector<unsigned int>& detIds,
                                         std::vector<double>& hitEnergies) {
  double ehcal(0);
  if (unCorrect_) {
    for (unsigned int k = 0; k < ids.size(); ++k) {
      double corr = (respCorrs->getValues(ids[k]))->getValue();
      if (corr != 0)
        edet[k] /= corr;
      ehcal += edet[k];
    }
  } else {
    for (const auto& en : edet)
      ehcal += en;
  }
  if ((std::abs(ehcal - eHcal) > 0.001) && (!unCorrect_))
    edm::LogWarning("HcalIsoTrack") << "Check inconsistent energies: " << eHcal << ":" << ehcal << " from "
                                    << ids.size() << " cells";
  eHcal = hcalScale_ * ehcal;

  if (collapseDepth_) {
    std::map<HcalDetId, double> hitMap;
    for (unsigned int k = 0; k < ids.size(); ++k) {
      HcalDetId id = hdc_->mergedDepthDetId(HcalDetId(ids[k]));
      auto itr = hitMap.find(id);
      if (itr == hitMap.end()) {
        hitMap[id] = edet[k];
      } else {
        (itr->second) += edet[k];
      }
    }
    detIds.reserve(hitMap.size());
    hitEnergies.reserve(hitMap.size());
    for (const auto& hit : hitMap) {
      detIds.emplace_back(hit.first.rawId());
      hitEnergies.emplace_back(hit.second);
    }
  } else {
    detIds.reserve(ids.size());
    hitEnergies.reserve(ids.size());
    for (unsigned int k = 0; k < ids.size(); ++k) {
      detIds.emplace_back(ids[k].rawId());
      hitEnergies.emplace_back(edet[k]);
    }
  }
#ifdef EDM_ML_DEBUG
  if (debug_) {
    edm::LogVerbatim("HcalIsoTrack") << "StoreEnergy1::Input to storeEnergy with " << ids.size() << " cells";
    for (unsigned int k = 0; k < ids.size(); ++k)
      edm::LogVerbatim("HcalIsoTrack") << "Hit [" << k << "] " << HcalDetId(ids[k]) << " E " << edet[k];
    edm::LogVerbatim("HcalIsoTrack") << "StoreEnergy1::Output of storeEnergy with " << detIds.size()
                                     << " cells and Etot " << eHcal;
    for (unsigned int k = 0; k < detIds.size(); ++k)
      edm::LogVerbatim("HcalIsoTrack") << "Hit [" << k << "] " << HcalDetId(detIds[k]) << " E " << hitEnergies[k];
  }
#endif
}

std::pair<double, double> AlCaHcalIsotrkProducer::storeEnergy(const HcalRespCorrs* respCorrs,
                                                              edm::Handle<HBHERecHitCollection>& hbhe,
                                                              const std::vector<DetId>& ids,
                                                              std::vector<double>& hitEnergy1,
                                                              std::vector<double>& hitEnergy2) {
  double ehcal1(0), ehcal2(0);
  std::vector<double> edet1, edet2;
  for (unsigned int k = 0; k < ids.size(); ++k) {
    double e1(0), e2(0);
    for (auto itr = hbhe->begin(); itr != hbhe->end(); ++itr) {
      if (itr->id() == ids[k]) {
        e1 = itr->eraw();
        e2 = itr->eaux();
        break;
      }
    }
    if (e1 < 1.e-10)
      e1 = 0;
    if (e2 < 1.e-10)
      e2 = 0;
    edet1.emplace_back(e1);
    edet2.emplace_back(e2);
  }
  if (unCorrect_) {
    for (unsigned int k = 0; k < ids.size(); ++k) {
      double corr = (respCorrs->getValues(ids[k]))->getValue();
      if (corr != 0) {
        edet1[k] /= corr;
        edet2[k] /= corr;
      }
    }
  }
  for (unsigned int k = 0; k < ids.size(); ++k) {
    ehcal1 += edet1[k];
    ehcal2 += edet2[k];
  }
  ehcal1 *= hcalScale_;
  ehcal2 *= hcalScale_;

  if (collapseDepth_) {
    std::map<HcalDetId, std::pair<double, double>> hitMap;
    for (unsigned int k = 0; k < ids.size(); ++k) {
      HcalDetId id = hdc_->mergedDepthDetId(HcalDetId(ids[k]));
      auto itr = hitMap.find(id);
      if (itr == hitMap.end()) {
        hitMap[id] = std::make_pair(edet1[k], edet2[k]);
      } else {
        (itr->second).first += edet1[k];
        (itr->second).second += edet2[k];
      }
    }
    hitEnergy1.reserve(hitMap.size());
    hitEnergy2.reserve(hitMap.size());
    for (const auto& hit : hitMap) {
      hitEnergy1.emplace_back(hit.second.first);
      hitEnergy2.emplace_back(hit.second.second);
    }
  } else {
    hitEnergy1.reserve(ids.size());
    hitEnergy2.reserve(ids.size());
    for (unsigned int k = 0; k < ids.size(); ++k) {
      hitEnergy1.emplace_back(edet1[k]);
      hitEnergy2.emplace_back(edet2[k]);
    }
  }
#ifdef EDM_ML_DEBUG
  if (debug_) {
    edm::LogVerbatim("HcalIsoTrack") << "StoreEnergy2::Input to storeEnergy with " << ids.size() << " cells";
    edm::LogVerbatim("HcalIsoTrack") << "StoreEnergy2::Output of storeEnergy with " << hitEnergy1.size()
                                     << " cells and Etot " << ehcal1 << ":" << ehcal2;
    for (unsigned int k = 0; k < hitEnergy1.size(); ++k)
      edm::LogVerbatim("HcalIsoTrack") << "Hit [" << k << "] " << hitEnergy1[k] << " : " << hitEnergy2[k];
  }
#endif
  return std::make_pair(ehcal1, ehcal2);
}

bool AlCaHcalIsotrkProducer::notaMuon(const reco::Track* pTrack0, const edm::Handle<reco::MuonCollection>& muonh) {
  bool flag(true);
  for (reco::MuonCollection::const_iterator recMuon = muonh->begin(); recMuon != muonh->end(); ++recMuon) {
    if (recMuon->innerTrack().isNonnull()) {
      const reco::Track* pTrack = (recMuon->innerTrack()).get();
      bool mediumMuon = (((recMuon->isPFMuon()) && (recMuon->isGlobalMuon() || recMuon->isTrackerMuon())) &&
                         (recMuon->innerTrack()->validFraction() > 0.49));
      if (mediumMuon) {
        double chiGlobal = ((recMuon->globalTrack().isNonnull()) ? recMuon->globalTrack()->normalizedChi2() : 999);
        bool goodGlob = (recMuon->isGlobalMuon() && chiGlobal < 3 &&
                         recMuon->combinedQuality().chi2LocalPosition < 12 && recMuon->combinedQuality().trkKink < 20);
        mediumMuon = muon::segmentCompatibility(*recMuon) > (goodGlob ? 0.303 : 0.451);
      }
      if (mediumMuon) {
        double dR = reco::deltaR(pTrack->eta(), pTrack->phi(), pTrack0->eta(), pTrack0->phi());
        if (dR < 0.1) {
          flag = false;
          break;
        }
      }
    }
  }
  return flag;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHcalIsotrkProducer);
