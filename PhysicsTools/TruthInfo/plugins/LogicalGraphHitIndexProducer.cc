// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/DetIdRecHitMap.h"

namespace {

  struct LogicalGraphView {
    explicit LogicalGraphView(truth::Graph const& graph) : graph_(graph) {}

    uint32_t nParticles() const { return graph_.nParticles(); }

    bool particleHasSim(uint32_t particleId) const {
      return particleId < graph_.particles().size() && graph_.particles()[particleId].hasSim();
    }

    int32_t particleSimNode(uint32_t particleId) const { return graph_.particles()[particleId].simNode; }

    template <typename F>
    void forEachParticleChild(uint32_t parentParticleId, F&& f) const {
      if (parentParticleId >= graph_.nParticles())
        return;

      for (const uint32_t vertexId : graph_.decayVertices(parentParticleId)) {
        if (vertexId >= graph_.nVertices())
          continue;

        for (const uint32_t childId : graph_.outgoingParticles(vertexId)) {
          f(childId);
        }
      }
    }

    truth::Graph const& graph_;
  };

  uint32_t checkedTrackId(int64_t key) {
    if (key < 0 || key > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      return 0;

    return static_cast<uint32_t>(key);
  }

  // Map a config channel name to its HitChannel; false if unknown.
  bool channelFromName(std::string const& name, truth::HitChannel& out) {
    if (name == "HGCalCalo") {
      out = truth::HitChannel::HGCalCalo;
      return true;
    }
    if (name == "Tracker") {
      out = truth::HitChannel::Tracker;
      return true;
    }
    if (name == "MTD") {
      out = truth::HitChannel::MTD;
      return true;
    }
    if (name == "Muon") {
      out = truth::HitChannel::Muon;
      return true;
    }
    return false;
  }

  bool inputTagLooksLikeHGCal(edm::InputTag const& tag) {
    const std::string& instance = tag.instance();
    return instance.find("HGCHits") != std::string::npos || instance.find("HGCEE") != std::string::npos ||
           instance.find("HGCHE") != std::string::npos;
  }

  bool inputTagLooksLikeHcal(edm::InputTag const& tag) {
    const std::string& instance = tag.instance();
    return instance.find("HcalHits") != std::string::npos || instance.find("Hcal") != std::string::npos;
  }

  struct RelabelContext {
    int geometryType = -1;

    std::array<HGCalTopology const*, 3> hgTopologies = {nullptr, nullptr, nullptr};
    std::array<HGCalDDDConstants const*, 3> hgConstants = {nullptr, nullptr, nullptr};

    HcalDDDRecConstants const* hcalConstants = nullptr;
  };

}  // namespace

class TruthLogicalGraphHitIndexProducer : public edm::global::EDProducer<> {
public:
  explicit TruthLogicalGraphHitIndexProducer(edm::ParameterSet const& cfg);
  ~TruthLogicalGraphHitIndexProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  void fillTrackToParticleMap(LogicalGraphView const& graph,
                              TruthGraph const& rawGraph,
                              truth::LogicalGraphHitIndexBuilder& builder) const;

  void fillSimHits(edm::Event& event,
                   edm::EventSetup const& setup,
                   truth::LogicalGraphHitIndexBuilder& builder,
                   hgcal::DetIdRecHitMap const* recHitMap) const;

  void fillTrackerSimHits(edm::Event& event, truth::LogicalGraphHitIndexBuilder& builder) const;

  // Muon chambers (DT/CSC/RPC/GEM/ME0): PSimHits keyed by trackId, like the tracker
  // channel (energy = energyLoss, no recHit link).
  void fillMuonSimHits(edm::Event& event, truth::LogicalGraphHitIndexBuilder& builder) const;

  // MTD (BTL/ETL): fill the MTD channel from the trackId-keyed MtdSimLayerClusters,
  // restricted to the signal interaction (the logical graph is signal-only).
  void fillMtdHits(edm::Event& event, truth::LogicalGraphHitIndexBuilder& builder) const;

  RelabelContext makeRelabelContext(edm::EventSetup const& setup) const;

  uint32_t recoDetIdForSimHit(PCaloHit const& simHit,
                              bool isHGCalCollection,
                              bool isHcalCollection,
                              RelabelContext const& context) const;

  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<TruthGraph> rawGraphToken_;
  edm::EDGetTokenT<hgcal::DetIdRecHitMap> recHitMapToken_;

  std::vector<edm::InputTag> simHitTags_;
  std::vector<edm::EDGetTokenT<std::vector<PCaloHit>>> simHitTokens_;

  std::vector<edm::InputTag> trackerSimHitTags_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> trackerSimHitTokens_;

  std::vector<edm::InputTag> muonSimHitTags_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> muonSimHitTokens_;

  edm::EDGetTokenT<MtdSimLayerClusterCollection> mtdSimLayerClusterToken_;
  edm::EDGetTokenT<MtdSimLayerClusterToRecoClusterAssociationMap> mtdSimToRecoAssocToken_;
  edm::EDGetTokenT<FTLClusterCollection> mtdBarrelClusterToken_;
  edm::EDGetTokenT<FTLClusterCollection> mtdEndcapClusterToken_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

  std::array<bool, truth::kNumHitChannels> fillChannel_{};

  bool doHGCalRelabelling_ = true;
};

TruthLogicalGraphHitIndexProducer::TruthLogicalGraphHitIndexProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawGraphToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      recHitMapToken_(consumes<hgcal::DetIdRecHitMap>(cfg.getParameter<edm::InputTag>("recHitMap"))),
      simHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("simHitCollections")),
      trackerSimHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("trackerSimHitCollections")),
      muonSimHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("muonSimHitCollections")),
      geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      doHGCalRelabelling_(cfg.getParameter<bool>("doHGCalRelabelling")) {
  simHitTokens_.reserve(simHitTags_.size());
  for (auto const& tag : simHitTags_) {
    simHitTokens_.push_back(consumes<std::vector<PCaloHit>>(tag));
  }

  trackerSimHitTokens_.reserve(trackerSimHitTags_.size());
  for (auto const& tag : trackerSimHitTags_) {
    trackerSimHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  }

  muonSimHitTokens_.reserve(muonSimHitTags_.size());
  for (auto const& tag : muonSimHitTags_) {
    muonSimHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  }

  mtdSimLayerClusterToken_ =
      consumes<MtdSimLayerClusterCollection>(cfg.getParameter<edm::InputTag>("mtdSimLayerClusters"));
  mtdSimToRecoAssocToken_ = consumes<MtdSimLayerClusterToRecoClusterAssociationMap>(
      cfg.getParameter<edm::InputTag>("mtdRecoClusterAssociation"));
  mtdBarrelClusterToken_ = consumes<FTLClusterCollection>(cfg.getParameter<edm::InputTag>("mtdBarrelClusters"));
  mtdEndcapClusterToken_ = consumes<FTLClusterCollection>(cfg.getParameter<edm::InputTag>("mtdEndcapClusters"));

  for (auto const& name : cfg.getParameter<std::vector<std::string>>("subdetectors")) {
    truth::HitChannel channel;
    if (channelFromName(name, channel))
      fillChannel_[static_cast<std::size_t>(channel)] = true;
    else
      edm::LogWarning("TruthLogicalGraphHitIndexProducer")
          << "Unknown subdetector channel '" << name << "'; ignoring it.";
  }

  produces<truth::LogicalGraphHitIndex>();
}

void TruthLogicalGraphHitIndexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));
  desc.add<edm::InputTag>("recHitMap", edm::InputTag("detIdToRecHitMapProducer"));

  desc.add<std::vector<std::string>>("subdetectors", {"HGCalCalo", "Tracker", "MTD", "Muon"})
      ->setComment(
          "Detector channels to fill (subdetector selection): any of HGCalCalo, Tracker, MTD, Muon. Each reads its "
          "own per-subdetector hit collections below; channels left out of this list stay empty in the index.");

  desc.add<std::vector<edm::InputTag>>("simHitCollections",
                                       {edm::InputTag("g4SimHits", "HGCHitsEE"),
                                        edm::InputTag("g4SimHits", "HGCHitsHEfront"),
                                        edm::InputTag("g4SimHits", "HGCHitsHEback")});

  desc.add<std::vector<edm::InputTag>>("trackerSimHitCollections",
                                       {edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTOBLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTECLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsTECHighTof")})
      ->setComment("Tracker PSimHit collections matched to particles via PSimHit::trackId()");

  desc.add<std::vector<edm::InputTag>>("muonSimHitCollections",
                                       {edm::InputTag("g4SimHits", "MuonDTHits"),
                                        edm::InputTag("g4SimHits", "MuonCSCHits"),
                                        edm::InputTag("g4SimHits", "MuonRPCHits"),
                                        edm::InputTag("g4SimHits", "MuonGEMHits"),
                                        edm::InputTag("g4SimHits", "MuonME0Hits")})
      ->setComment("Muon-chamber PSimHit collections matched to particles via PSimHit::trackId()");

  desc.add<bool>("doHGCalRelabelling", true)
      ->setComment("Convert old HGCAL simulation DetIds to reco DetIds before looking up recHits");

  desc.add<edm::InputTag>("mtdSimLayerClusters", edm::InputTag("mix", "MergedMtdTruthLC"))
      ->setComment(
          "MtdSimLayerCluster collection (BTL/ETL); keyed by SimTrack trackId via particleId(). The signal "
          "interaction is selected by EncodedEventId; pile-up clusters are skipped.");
  desc.add<edm::InputTag>("mtdRecoClusterAssociation", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation"))
      ->setComment("MtdSimLayerCluster -> FTLCluster association; sets the MTD recHitIndex when available.");
  desc.add<edm::InputTag>("mtdBarrelClusters", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("mtdEndcapClusters", edm::InputTag("mtdClusters", "FTLEndcap"))
      ->setComment(
          "Reco FTLClusters; the MTD recHitIndex is the global index in the barrel-then-endcap concatenation.");

  descriptions.addWithDefaultLabel(desc);
}

void TruthLogicalGraphHitIndexProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const {
  auto const& graph = event.get(graphToken_);
  auto const& rawGraph = event.get(rawGraphToken_);

  edm::Handle<hgcal::DetIdRecHitMap> hRecHitMap;
  event.getByToken(recHitMapToken_, hRecHitMap);
  auto const* recHitMap = hRecHitMap.isValid() ? &(*hRecHitMap) : nullptr;

  LogicalGraphView graphView(graph);

  truth::LogicalGraphHitIndexBuilder builder(graphView.nParticles());

  fillTrackToParticleMap(graphView, rawGraph, builder);

  // Each subdetector channel is filled only when selected (see "subdetectors").
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::HGCalCalo)])
    fillSimHits(event, setup, builder, recHitMap);
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::Tracker)])
    fillTrackerSimHits(event, builder);
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::Muon)])
    fillMuonSimHits(event, builder);
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::MTD)])
    fillMtdHits(event, builder);

  auto output = std::make_unique<truth::LogicalGraphHitIndex>(builder.finish());
  event.put(std::move(output));
}

void TruthLogicalGraphHitIndexProducer::fillTrackToParticleMap(LogicalGraphView const& graph,
                                                               TruthGraph const& rawGraph,
                                                               truth::LogicalGraphHitIndexBuilder& builder) const {
  for (uint32_t particleId = 0; particleId < graph.nParticles(); ++particleId) {
    if (!graph.particleHasSim(particleId))
      continue;

    const int32_t simNode = graph.particleSimNode(particleId);
    if (simNode < 0)
      continue;

    const uint32_t simNodeU32 = static_cast<uint32_t>(simNode);
    if (simNodeU32 >= rawGraph.nNodes())
      continue;

    auto const& ref = rawGraph.nodeRef(simNodeU32);
    if (ref.kind != TruthGraph::NodeKind::SimTrack)
      continue;

    const uint32_t trackId = checkedTrackId(ref.key);
    if (trackId == 0)
      continue;

    builder.setSimTrackForParticle(particleId, trackId);
  }

  for (uint32_t parentId = 0; parentId < graph.nParticles(); ++parentId) {
    graph.forEachParticleChild(parentId, [&](uint32_t childId) { builder.addParticleChild(parentId, childId); });
  }
}

RelabelContext TruthLogicalGraphHitIndexProducer::makeRelabelContext(edm::EventSetup const& setup) const {
  RelabelContext context;

  if (!doHGCalRelabelling_)
    return context;

  auto const& geom = setup.getData(geomToken_);

  auto const* hcalGeometry = static_cast<HcalGeometry const*>(geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
  if (hcalGeometry != nullptr) {
    context.hcalConstants = hcalGeometry->topology().dddConstants();
  }

  auto const* eeGeometry =
      static_cast<HGCalGeometry const*>(geom.getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty));

  if (eeGeometry != nullptr) {
    context.geometryType = 1;

    auto const* fhGeometry = static_cast<HGCalGeometry const*>(
        geom.getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
    auto const* bhGeometry = static_cast<HGCalGeometry const*>(
        geom.getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));

    context.hgTopologies[0] = &eeGeometry->topology();
    context.hgTopologies[1] = fhGeometry != nullptr ? &fhGeometry->topology() : nullptr;
    context.hgTopologies[2] = bhGeometry != nullptr ? &bhGeometry->topology() : nullptr;

    for (unsigned i = 0; i < context.hgTopologies.size(); ++i) {
      if (context.hgTopologies[i] != nullptr)
        context.hgConstants[i] = &context.hgTopologies[i]->dddConstants();
    }

    return context;
  }

  context.geometryType = 0;

  eeGeometry = static_cast<HGCalGeometry const*>(geom.getSubdetectorGeometry(DetId::Forward, HGCEE));
  auto const* fhGeometry = static_cast<HGCalGeometry const*>(geom.getSubdetectorGeometry(DetId::Forward, HGCHEF));

  context.hgTopologies[0] = eeGeometry != nullptr ? &eeGeometry->topology() : nullptr;
  context.hgTopologies[1] = fhGeometry != nullptr ? &fhGeometry->topology() : nullptr;

  for (unsigned i = 0; i < context.hgTopologies.size(); ++i) {
    if (context.hgTopologies[i] != nullptr)
      context.hgConstants[i] = &context.hgTopologies[i]->dddConstants();
  }

  return context;
}

uint32_t TruthLogicalGraphHitIndexProducer::recoDetIdForSimHit(PCaloHit const& simHit,
                                                               bool isHGCalCollection,
                                                               bool isHcalCollection,
                                                               RelabelContext const& context) const {
  const uint32_t simId = simHit.id();

  if (!doHGCalRelabelling_) {
    return simId;
  }

  if (isHGCalCollection) {
    if (context.geometryType == 1) {
      return simId;
    }

    int subdet = 0;
    int layer = 0;
    int cell = 0;
    int sec = 0;
    int subsec = 0;
    int zp = 0;

    HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);

    const int hgcalIndex = subdet - 3;
    if (hgcalIndex < 0 || hgcalIndex >= static_cast<int>(context.hgConstants.size()))
      return 0;

    auto const* constants = context.hgConstants[hgcalIndex];
    auto const* topology = context.hgTopologies[hgcalIndex];

    if (constants == nullptr || topology == nullptr)
      return 0;

    const auto recoLayerCell = constants->simToReco(cell, layer, sec, topology->detectorType());
    cell = recoLayerCell.first;
    layer = recoLayerCell.second;

    if (layer < 0)
      return 0;

    return HGCalDetId(static_cast<ForwardSubdetector>(subdet), zp, layer, subsec, sec, cell).rawId();
  }

  if (isHcalCollection && context.hcalConstants != nullptr) {
    return HcalHitRelabeller::relabel(simId, context.hcalConstants).rawId();
  }

  return simId;
}

void TruthLogicalGraphHitIndexProducer::fillSimHits(edm::Event& event,
                                                    edm::EventSetup const& setup,
                                                    truth::LogicalGraphHitIndexBuilder& builder,
                                                    hgcal::DetIdRecHitMap const* recHitMap) const {
  const RelabelContext relabelContext = makeRelabelContext(setup);

  for (uint32_t tokenIndex = 0; tokenIndex < simHitTokens_.size(); ++tokenIndex) {
    auto const& token = simHitTokens_[tokenIndex];
    auto const& tag = simHitTags_[tokenIndex];

    edm::Handle<std::vector<PCaloHit>> hSimHits;
    event.getByToken(token, hSimHits);

    if (!hSimHits.isValid()) {
      edm::LogWarning("TruthLogicalGraphHitIndexProducer")
          << "Missing PCaloHit collection " << tag.encode() << ". Skipping it.";
      continue;
    }

    const bool isHGCalCollection = inputTagLooksLikeHGCal(tag);
    const bool isHcalCollection = inputTagLooksLikeHcal(tag);

    for (auto const& simHit : *hSimHits) {
      const int geantTrackId = simHit.geantTrackId();
      if (geantTrackId <= 0)
        continue;

      const uint32_t detId = recoDetIdForSimHit(simHit, isHGCalCollection, isHcalCollection, relabelContext);
      if (detId == 0)
        continue;

      uint32_t recHitIndex = truth::LogicalGraphHitIndex::Hit::kInvalidRecHitIndex;

      if (recHitMap != nullptr) {
        const auto it = recHitMap->find(detId);
        if (it != recHitMap->end()) {
          recHitIndex = it->second;
        }
      }

      builder.addHit(
          truth::HitChannel::HGCalCalo, static_cast<uint32_t>(geantTrackId), detId, simHit.energy(), recHitIndex);
    }
  }
}

void TruthLogicalGraphHitIndexProducer::fillTrackerSimHits(edm::Event& event,
                                                           truth::LogicalGraphHitIndexBuilder& builder) const {
  for (uint32_t tokenIndex = 0; tokenIndex < trackerSimHitTokens_.size(); ++tokenIndex) {
    edm::Handle<edm::PSimHitContainer> hSimHits;
    event.getByToken(trackerSimHitTokens_[tokenIndex], hSimHits);

    if (!hSimHits.isValid()) {
      edm::LogWarning("TruthLogicalGraphHitIndexProducer")
          << "Missing tracker PSimHit collection " << trackerSimHitTags_[tokenIndex].encode() << ". Skipping it.";
      continue;
    }

    for (auto const& simHit : *hSimHits) {
      // PSimHit::trackId() is the G4 trackId of the SimTrack that made the hit,
      // the same id space used to associate calorimeter simhits to particles.
      builder.addHit(truth::HitChannel::Tracker, simHit.trackId(), simHit.detUnitId(), simHit.energyLoss());
    }
  }
}

void TruthLogicalGraphHitIndexProducer::fillMuonSimHits(edm::Event& event,
                                                        truth::LogicalGraphHitIndexBuilder& builder) const {
  for (uint32_t tokenIndex = 0; tokenIndex < muonSimHitTokens_.size(); ++tokenIndex) {
    edm::Handle<edm::PSimHitContainer> hSimHits;
    event.getByToken(muonSimHitTokens_[tokenIndex], hSimHits);

    // Phase-2 D120 does not populate every muon subsystem; missing ones are skipped.
    if (!hSimHits.isValid())
      continue;

    for (auto const& simHit : *hSimHits) {
      builder.addHit(truth::HitChannel::Muon, simHit.trackId(), simHit.detUnitId(), simHit.energyLoss());
    }
  }
}

void TruthLogicalGraphHitIndexProducer::fillMtdHits(edm::Event& event,
                                                    truth::LogicalGraphHitIndexBuilder& builder) const {
  edm::Handle<MtdSimLayerClusterCollection> hClusters;
  event.getByToken(mtdSimLayerClusterToken_, hClusters);
  if (!hClusters.isValid())
    return;

  // Optional reco-cluster link: the MtdSimLayerCluster -> FTLCluster association plus
  // the two FTLCluster collections give the MTD recHitIndex as the global index in the
  // barrel-then-endcap concatenation. Absent (e.g. no MTD reco) -> recHitIndex invalid.
  edm::Handle<MtdSimLayerClusterToRecoClusterAssociationMap> hAssoc;
  event.getByToken(mtdSimToRecoAssocToken_, hAssoc);
  edm::Handle<FTLClusterCollection> hBarrel;
  event.getByToken(mtdBarrelClusterToken_, hBarrel);
  edm::Handle<FTLClusterCollection> hEndcap;
  event.getByToken(mtdEndcapClusterToken_, hEndcap);
  const bool haveReco = hAssoc.isValid() && hBarrel.isValid() && hEndcap.isValid();
  const uint32_t nBarrelClusters = haveReco ? static_cast<uint32_t>(hBarrel->dataSize()) : 0;

  for (uint32_t i = 0; i < hClusters->size(); ++i) {
    auto const& cluster = (*hClusters)[i];

    // Only the signal interaction (bx 0, event 0): the logical graph is signal-only,
    // so its trackId space matches the signal MtdSimLayerClusters; pile-up clusters
    // (different EncodedEventId) could collide numerically and are skipped.
    const EncodedEventId eid = cluster.eventId();
    if (eid.bunchCrossing() != 0 || eid.event() != 0)
      continue;

    // The best-matched reco FTLCluster -> a global index across the two collections.
    uint32_t recHitIndex = truth::LogicalGraphHitIndex::Hit::kInvalidRecHitIndex;
    if (haveReco) {
      const MtdSimLayerClusterRef simRef(hClusters, i);
      const auto range = hAssoc->equal_range(simRef);
      if (range.first != range.second && !range.first->second.empty()) {
        FTLClusterRef const& recoRef = range.first->second.front();
        if (recoRef.id() == hBarrel.id())
          recHitIndex = static_cast<uint32_t>(recoRef.key());
        else if (recoRef.id() == hEndcap.id())
          recHitIndex = nBarrelClusters + static_cast<uint32_t>(recoRef.key());
      }
    }

    // particleId() carries the producing SimTrack trackId; hits_and_energies() returns
    // (packed sensor-module DetId << 32 | row << 16 | col, energy). The builder
    // coalesces per module DetId; every hit of the cluster shares the matched FTLCluster.
    const auto trackId = static_cast<uint32_t>(cluster.particleId());
    for (auto const& [packedHit, energy] : cluster.hits_and_energies()) {
      const uint32_t moduleDetId = static_cast<uint32_t>(packedHit >> 32);
      builder.addHit(truth::HitChannel::MTD, trackId, moduleDetId, energy, recHitIndex);
    }
  }
}

DEFINE_FWK_MODULE(TruthLogicalGraphHitIndexProducer);
