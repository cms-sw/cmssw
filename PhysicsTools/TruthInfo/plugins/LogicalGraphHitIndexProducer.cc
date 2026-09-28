// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/SimHitCategory.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"
#include "PhysicsTools/TruthInfo/interface/TrackerCells.h"
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
    if (name == "Calo") {
      out = truth::HitChannel::Calo;
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

  void fillTrackerCells(edm::Event& event, truth::LogicalGraphHitIndexBuilder& builder) const;

  // Muon chambers (DT/CSC/RPC/GEM/ME0): PSimHits keyed by trackId, like the tracker
  // channel (energy = energyLoss, no recHit link).
  void fillMuonSimHits(edm::Event& event, truth::LogicalGraphHitIndexBuilder& builder) const;

  // MTD (BTL/ETL): fill the MTD channel from the MtdSimLayerClusters of every
  // interaction, one hit per (sensor module, cell, category) with its earliest time.
  void fillMtdHits(edm::Event& event, edm::EventSetup const& setup, truth::LogicalGraphHitIndexBuilder& builder) const;

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

  // Per-cell truth of the tracker, written by the digitizer: (channel, trackId,
  // eventId, charge fraction).
  std::vector<edm::InputTag> digiSimLinkTags_;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>>> digiSimLinkTokens_;
  // One warning per job per collection that is missing from the input.
  mutable std::vector<std::once_flag> digiSimLinkWarned_;
  std::vector<edm::InputTag> muonSimHitTags_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> muonSimHitTokens_;

  edm::EDGetTokenT<MtdSimLayerClusterCollection> mtdSimLayerClusterToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdTopologyToken_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

  std::array<bool, truth::kNumHitChannels> fillChannel_{};

  // Sim-to-reco DetId conversion, one switch per calorimeter numbering scheme: the
  // HGCAL hexagon unpacking and the HCAL HcalHitRelabeller are selected independently.
  bool doHGCalRelabelling_ = true;
  bool doHcalRelabelling_ = true;
  bool sharedSubgraphStore_ = false;
};

TruthLogicalGraphHitIndexProducer::TruthLogicalGraphHitIndexProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawGraphToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      recHitMapToken_(consumes<hgcal::DetIdRecHitMap>(cfg.getParameter<edm::InputTag>("recHitMap"))),
      simHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("simHitCollections")),
      digiSimLinkTags_(cfg.getParameter<std::vector<edm::InputTag>>("trackerDigiSimLinks")),
      muonSimHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("muonSimHitCollections")),
      geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      doHGCalRelabelling_(cfg.getParameter<bool>("doHGCalRelabelling")),
      doHcalRelabelling_(cfg.getParameter<bool>("doHcalRelabelling")),
      sharedSubgraphStore_(cfg.getParameter<bool>("sharedSubgraphStore")) {
  simHitTokens_.reserve(simHitTags_.size());
  for (auto const& tag : simHitTags_) {
    simHitTokens_.push_back(consumes<std::vector<PCaloHit>>(tag));
  }

  digiSimLinkTokens_.reserve(digiSimLinkTags_.size());
  for (auto const& tag : digiSimLinkTags_) {
    digiSimLinkTokens_.push_back(consumes<edm::DetSetVector<PixelDigiSimLink>>(tag));
  }
  digiSimLinkWarned_ = std::vector<std::once_flag>(digiSimLinkTags_.size());

  muonSimHitTokens_.reserve(muonSimHitTags_.size());
  for (auto const& tag : muonSimHitTags_) {
    muonSimHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  }

  mtdSimLayerClusterToken_ =
      consumes<MtdSimLayerClusterCollection>(cfg.getParameter<edm::InputTag>("mtdSimLayerClusters"));

  for (auto const& name : cfg.getParameter<std::vector<std::string>>("subdetectors")) {
    truth::HitChannel channel;
    if (channelFromName(name, channel))
      fillChannel_[static_cast<std::size_t>(channel)] = true;
    else
      edm::LogWarning("TruthLogicalGraphHitIndexProducer")
          << "Unknown subdetector channel '" << name << "'; ignoring it.";
  }
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::MTD)])
    mtdTopologyToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();

  produces<truth::LogicalGraphHitIndex>();
}

void TruthLogicalGraphHitIndexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));
  desc.add<edm::InputTag>("recHitMap", edm::InputTag("detIdToRecHitMapProducer"));

  desc.add<std::vector<std::string>>("subdetectors", {"Calo", "Tracker", "MTD", "Muon"})
      ->setComment(
          "Detector channels to fill (subdetector selection): any of Calo, Tracker, MTD, Muon. Each reads its "
          "own per-subdetector hit collections below; channels left out of this list stay empty in the index.");

  // The same calorimeter list TruthLogicalGraphProducer prunes on. The pruner deletes
  // any SIM particle whose subgraph carries no hit in the collections IT reads, so a
  // shorter list here leaves a kept particle with an empty footprint: a barrel particle
  // would show zero calorimeter hits and every per-cell fraction over it would be wrong.
  // Keep the two defaults equal.
  desc.add<std::vector<edm::InputTag>>("simHitCollections",
                                       {edm::InputTag("g4SimHits", "HGCHitsEE"),
                                        edm::InputTag("g4SimHits", "HGCHitsHEfront"),
                                        edm::InputTag("g4SimHits", "HGCHitsHEback"),
                                        edm::InputTag("g4SimHits", "EcalHitsEB"),
                                        edm::InputTag("g4SimHits", "HcalHits")});

  desc.add<std::vector<edm::InputTag>>(
          "trackerDigiSimLinks",
          {edm::InputTag("simSiPixelDigis", "Pixel"), edm::InputTag("simSiPixelDigis", "Tracker")})
      ->setComment(
          "Digi sim links of the tracker, the inner one and the outer one. The tracker truth is keyed by "
          "(module, cell) and comes from these alone, which is what separates two particles crossing one module. "
          "A module no links product covers carries no tracker truth");
  desc.add<std::vector<edm::InputTag>>("muonSimHitCollections",
                                       {edm::InputTag("g4SimHits", "MuonDTHits"),
                                        edm::InputTag("g4SimHits", "MuonCSCHits"),
                                        edm::InputTag("g4SimHits", "MuonRPCHits"),
                                        edm::InputTag("g4SimHits", "MuonGEMHits"),
                                        edm::InputTag("g4SimHits", "MuonME0Hits")})
      ->setComment("Muon-chamber PSimHit collections matched to particles via PSimHit::trackId()");

  desc.add<bool>("doHGCalRelabelling", true)
      ->setComment(
          "Convert old HGCAL simulation DetIds to reco DetIds before looking up recHits. No-op for the Run4 "
          "geometries, whose HGCAL simulation DetId is already the reco DetId. Affects the HGCAL collections only.");
  desc.add<bool>("doHcalRelabelling", true)
      ->setComment(
          "Apply HcalHitRelabeller to the HCAL simulation DetIds, which are in packed test numbering, so the index "
          "stores the reco HcalDetIds the association matches on. Affects the HCAL collections only.");

  desc.add<bool>("sharedSubgraphStore", true)
      ->setComment(
          "Store each hit once, ordered so that a particle's subtree is a contiguous range, instead of copying every "
          "descendant's hits into each ancestor's aggregate, which cuts the index from 445 to 202 kB/event on ttbar. "
          "LogicalGraphHitIndex::subgraphHits returns a single span, so under this layout it is empty for a "
          "GEN-only particle, whose subgraph spans several ranges: a consumer that can be handed any particle uses "
          "truth::SubgraphHitView, which is correct in both layouts. Set false to write the materialised layout. "
          "Reading either layout is automatic.");

  desc.add<edm::InputTag>("mtdSimLayerClusters", edm::InputTag("mix", "MergedMtdTruthLC"))
      ->setComment(
          "MtdSimLayerCluster collection (BTL/ETL), keyed by (EncodedEventId, SimTrack trackId). The MTD "
          "channel is cell keyed: detId is the sensor module, recHitIndex the cell as "
          "category << 24 | row << 16 | col. Bits 0 to 23 are the (row, col) of an FTLCluster pixel on "
          "that module, so a consumer masks the category before it compares with a reco pixel.");

  descriptions.addWithDefaultLabel(desc);
}

void TruthLogicalGraphHitIndexProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const {
  auto const& graph = event.get(graphToken_);
  auto const& rawGraph = event.get(rawGraphToken_);

  edm::Handle<hgcal::DetIdRecHitMap> hRecHitMap;
  event.getByToken(recHitMapToken_, hRecHitMap);
  auto const* recHitMap = hRecHitMap.isValid() ? &(*hRecHitMap) : nullptr;

  LogicalGraphView graphView(graph);

  truth::LogicalGraphHitIndexBuilder builder(graphView.nParticles(), sharedSubgraphStore_);

  fillTrackToParticleMap(graphView, rawGraph, builder);

  // Each subdetector channel is filled only when selected (see "subdetectors").
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::Calo)])
    fillSimHits(event, setup, builder, recHitMap);
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::Tracker)]) {
    // The tracker truth is keyed by (module, cell) and comes from the digi sim links
    // alone: a tracker DetId names a module, and the cell is what separates two
    // particles crossing one.
    builder.setCellKeyed(truth::HitChannel::Tracker, true);
    fillTrackerCells(event, builder);
  }
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::Muon)])
    fillMuonSimHits(event, builder);
  if (fillChannel_[static_cast<std::size_t>(truth::HitChannel::MTD)]) {
    builder.setCellKeyed(truth::HitChannel::MTD, true);
    fillMtdHits(event, setup, builder);
  }

  auto output = std::make_unique<truth::LogicalGraphHitIndex>(builder.finish());
  if (sharedSubgraphStore_ && !builder.usedSharedStore()) {
    // The materialised layout stores each hit once PER ANCESTOR, and on a large event
    // that can exceed ROOT's 1 GiB single-object limit and kill the output module. A
    // silent fallback shows up only as a crash three modules away, on heavy-ion events
    // (cms-sw/cmssw#51638), so the fallback is always announced.
    edm::LogWarning("LogicalGraphHitIndexProducer")
        << "shared subgraph store requested but the hit-carrying particles do not form a forest; "
           "fell back to the MATERIALISED layout, which duplicates every hit per ancestor. On a "
           "high-multiplicity event this can exceed ROOT's 1 GiB single-object limit at output.";
  }
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

    builder.setSimTrackForParticle(particleId, rawGraph.nodeEventId(simNodeU32), trackId);
  }

  for (uint32_t parentId = 0; parentId < graph.nParticles(); ++parentId) {
    graph.forEachParticleChild(parentId, [&](uint32_t childId) { builder.addParticleChild(parentId, childId); });
  }
}

RelabelContext TruthLogicalGraphHitIndexProducer::makeRelabelContext(edm::EventSetup const& setup) const {
  RelabelContext context;

  if (!doHGCalRelabelling_ && !doHcalRelabelling_)
    return context;

  auto const& geom = setup.getData(geomToken_);

  if (doHcalRelabelling_) {
    auto const* hcalGeometry = static_cast<HcalGeometry const*>(geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
    if (hcalGeometry != nullptr) {
      context.hcalConstants = hcalGeometry->topology().dddConstants();
    }
  }

  if (!doHGCalRelabelling_)
    return context;

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

  if (isHcalCollection) {
    if (doHcalRelabelling_ && context.hcalConstants != nullptr)
      return HcalHitRelabeller::relabel(simId, context.hcalConstants).rawId();
    return simId;
  }

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

      builder.addHit(truth::HitChannel::Calo,
                     simHit.eventId().rawId(),
                     static_cast<uint32_t>(geantTrackId),
                     detId,
                     simHit.energy(),
                     recHitIndex);
    }
  }
}

void TruthLogicalGraphHitIndexProducer::fillTrackerCells(edm::Event& event,
                                                         truth::LogicalGraphHitIndexBuilder& builder) const {
  for (uint32_t tokenIndex = 0; tokenIndex < digiSimLinkTokens_.size(); ++tokenIndex) {
    edm::Handle<edm::DetSetVector<PixelDigiSimLink>> hLinks;
    event.getByToken(digiSimLinkTokens_[tokenIndex], hLinks);

    if (!hLinks.isValid()) {
      std::call_once(digiSimLinkWarned_[tokenIndex], [this, tokenIndex]() {
        edm::LogWarning("TruthLogicalGraphHitIndexProducer")
            << "Missing digi sim links " << digiSimLinkTags_[tokenIndex].encode()
            << ". The modules they cover carry no tracker truth for this job.";
      });
      continue;
    }

    for (auto const& detSet : *hLinks) {
      for (auto const& link : detSet) {
        // The energy of a cell-keyed hit is the charge fraction the digitizer recorded
        // for this particle on this cell. The tracker metric counts cells, so the value
        // is informational, but it must be positive or the builder drops the hit.
        const float fraction = link.fraction() > 0.f ? link.fraction() : 1.f;
        builder.addHit(truth::HitChannel::Tracker,
                       link.eventId().rawId(),
                       link.SimTrackId(),
                       detSet.detId(),
                       fraction,
                       static_cast<uint32_t>(link.channel()));
      }
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
      builder.addHit(
          truth::HitChannel::Muon, simHit.eventId().rawId(), simHit.trackId(), simHit.detUnitId(), simHit.energyLoss());
    }
  }
}

void TruthLogicalGraphHitIndexProducer::fillMtdHits(edm::Event& event,
                                                    edm::EventSetup const& setup,
                                                    truth::LogicalGraphHitIndexBuilder& builder) const {
  edm::Handle<MtdSimLayerClusterCollection> hClusters;
  event.getByToken(mtdSimLayerClusterToken_, hClusters);
  if (!hClusters.isValid())
    return;

  // A BTL sim hit carries the DetId of its crystal, while a reco cluster carries the
  // sensor module; the crystal layout of the topology maps one to the other, as
  // MTDGeomUtil::sensorModuleId does. An ETL sim hit already names its module.
  const auto crystalLayout =
      MTDTopologyMode::crysLayoutFromTopoMode(setup.getData(mtdTopologyToken_).getMTDTopologyMode());

  for (auto const& cluster : *hClusters) {
    // Every interaction: the graph keys its particles by (EncodedEventId, trackId), as
    // the other channels do.
    const uint64_t eventId = cluster.eventId().rawId();
    const auto trackId = static_cast<uint32_t>(cluster.particleId());
    const uint32_t category = cluster.hitProdType();
    const auto energies = cluster.hits_and_energies();
    const auto times = cluster.hits_and_times();
    for (std::size_t i = 0; i < energies.size(); ++i) {
      // Packed as the sim DetId << 32 | row << 16 | col, the row and col are 8 bits each.
      const uint64_t packed = energies[i].first;
      const MTDDetId simId(static_cast<uint32_t>(packed >> 32));
      const uint32_t moduleId = simId.mtdSubDetector() == MTDDetId::BTL
                                    ? BTLDetId(simId.rawId()).geographicalId(crystalLayout).rawId()
                                    : ETLDetId(simId.rawId()).geographicalId().rawId();
      using Hit = truth::LogicalGraphHitIndex::Hit;
      static_assert(SimHitCategory::nCategoriesMTD <= (1u << (32 - Hit::kMtdCategoryShift)));
      const uint32_t cell = category << Hit::kMtdCategoryShift | (static_cast<uint32_t>(packed) & Hit::kMtdCellMask);
      builder.addTimedHit(
          truth::HitChannel::MTD, eventId, trackId, moduleId, energies[i].second, cell, times[i].second);
    }
  }
}

DEFINE_FWK_MODULE(TruthLogicalGraphHitIndexProducer);
