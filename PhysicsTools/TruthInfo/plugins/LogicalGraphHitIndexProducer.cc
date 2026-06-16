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

#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/DetIdRecHitMap.h"

namespace {

  struct LogicalGraphView {
    explicit LogicalGraphView(truth::Graph const& graph) : graph_(graph) {}

    uint32_t nParticles() const { return graph_.nParticles(); }

    bool particleHasSim(uint32_t particleId) const {
      return particleId < graph_.particles.size() && graph_.particles[particleId].hasSim();
    }

    int32_t particleSimNode(uint32_t particleId) const { return graph_.particles[particleId].simNode; }

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

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

  bool doHGCalRelabelling_ = true;
};

TruthLogicalGraphHitIndexProducer::TruthLogicalGraphHitIndexProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawGraphToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      recHitMapToken_(consumes<hgcal::DetIdRecHitMap>(cfg.getParameter<edm::InputTag>("recHitMap"))),
      simHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("simHitCollections")),
      trackerSimHitTags_(cfg.getParameter<std::vector<edm::InputTag>>("trackerSimHitCollections")),
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

  produces<truth::LogicalGraphHitIndex>();
}

void TruthLogicalGraphHitIndexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));
  desc.add<edm::InputTag>("recHitMap", edm::InputTag("simHitToRecHitMapProducer"));

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

  desc.add<bool>("doHGCalRelabelling", true)
      ->setComment("Convert old HGCAL simulation DetIds to reco DetIds before looking up recHits");

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
  fillSimHits(event, setup, builder, recHitMap);
  fillTrackerSimHits(event, builder);

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

      uint32_t recHitIndex = truth::LogicalGraphHitIndex::Hit::invalidRecHitIndex;

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

DEFINE_FWK_MODULE(TruthLogicalGraphHitIndexProducer);
