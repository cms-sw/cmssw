// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Phase-B (B1): build the mixed (signal + pileup) raw TruthGraph as a
// DigiAccumulatorMixMod, the way TrackingTruthAccumulator / CaloTruthAccumulator
// work. The framework hands us one sub-event at a time with its NATIVE
// SimTrack/SimVertex collections, so trackId/vertIndex/parentIndex are used in
// their original local context (no flattening, no cross-pileup keying) and the
// graph does not fragment the way the Phase-A MixCollection prototype did. It is
// also identical for standard mixing and premixing, and is consistent with the
// digis by construction (same sub-events).
//
// Configurable:
//   pileupBunchCrossings : which bunch crossings to include for pileup
//                          (default {0} = in-time pileup only).
//   collapsePileupGen    : for pileup, collapse the GEN decay chain and keep the
//                          stable GEN particles on their production vertex plus
//                          the SIM continuation (default true). [GEN enrichment is
//                          staged; this first cut accumulates the SIM realm, which
//                          is the "keep the sim" backbone, tagged per bunch
//                          crossing; the signal keeps its full graph via the
//                          standard TruthGraphProducer.]
//
// Each sub-event's nodes carry an EncodedEventId: (0,0) for the signal, and
// (bunchCrossing, pileupIndex) for pileup, so signal and pileup stay
// distinguishable downstream (Branch::isFromPileup / bunchCrossing).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {
  uint64_t packEventId(EncodedEventId const& id) {
    uint64_t out = 0;
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }
}  // namespace

class TruthGraphAccumulator : public DigiAccumulatorMixMod {
public:
  TruthGraphAccumulator(edm::ParameterSet const&, edm::ProducesCollector, edm::ConsumesCollector&);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;

private:
  // Append one sub-event's SIM realm to the accumulator, tagging every node with
  // `eid`. SimTrack/SimVertex ids are local to this sub-event.
  void addSimSubEvent(edm::SimTrackContainer const& tracks,
                      edm::SimVertexContainer const& vertices,
                      EncodedEventId const& eid);

  const edm::InputTag simTrackTag_;
  const edm::InputTag simVertexTag_;
  const std::vector<int> pileupBunchCrossings_;
  const bool collapsePileupGen_;

  // Per-bunch-crossing pileup counter, to give each pileup interaction a distinct
  // EncodedEventId event number.
  std::unordered_map<int, int> pileupCountByBx_;

  // Accumulated raw graph (reset in initializeEvent, emitted in finalizeEvent).
  std::vector<TruthGraph::NodeRef> nodes_;
  std::vector<int32_t> pdgId_;
  std::vector<uint64_t> eventId_;
  std::vector<int32_t> simTrackToVtx_;
  std::vector<std::pair<uint32_t, uint32_t>> edges_;
  std::vector<uint8_t> edgeKinds_;

  [[nodiscard]] bool keepBx(int bx) const {
    return std::find(pileupBunchCrossings_.begin(), pileupBunchCrossings_.end(), bx) != pileupBunchCrossings_.end();
  }
};

TruthGraphAccumulator::TruthGraphAccumulator(edm::ParameterSet const& cfg,
                                             edm::ProducesCollector producesCollector,
                                             edm::ConsumesCollector& iC)
    : simTrackTag_(cfg.getParameter<edm::InputTag>("simTracks")),
      simVertexTag_(cfg.getParameter<edm::InputTag>("simVertices")),
      pileupBunchCrossings_(cfg.getParameter<std::vector<int>>("pileupBunchCrossings")),
      collapsePileupGen_(cfg.getParameter<bool>("collapsePileupGen")) {
  producesCollector.produces<TruthGraph>();
  // Signal sub-event tokens (pileup sub-events are read by label via PileUpEventPrincipal).
  iC.consumes<edm::SimTrackContainer>(simTrackTag_);
  iC.consumes<edm::SimVertexContainer>(simVertexTag_);
}

void TruthGraphAccumulator::initializeEvent(edm::Event const&, edm::EventSetup const&) {
  pileupCountByBx_.clear();
  nodes_.clear();
  pdgId_.clear();
  eventId_.clear();
  simTrackToVtx_.clear();
  edges_.clear();
  edgeKinds_.clear();
}

void TruthGraphAccumulator::addSimSubEvent(edm::SimTrackContainer const& tracks,
                                           edm::SimVertexContainer const& vertices,
                                           EncodedEventId const& eid) {
  const uint64_t packed = packEventId(eid);
  const uint32_t baseVtx = static_cast<uint32_t>(nodes_.size());

  // Vertex nodes (local vector index = position from baseVtx). vertexId -> node.
  std::unordered_map<uint32_t, uint32_t> vertexIdToNode;
  vertexIdToNode.reserve(vertices.size() * 2);
  for (auto const& v : vertices) {
    const uint32_t node = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(TruthGraph::NodeRef{TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(v.vertexId())});
    pdgId_.push_back(0);
    eventId_.push_back(packed);
    simTrackToVtx_.push_back(-1);
    vertexIdToNode.emplace(static_cast<uint32_t>(v.vertexId()), node);
  }

  // Track nodes. trackId -> node.
  std::unordered_map<uint32_t, uint32_t> trackIdToNode;
  trackIdToNode.reserve(tracks.size() * 2);
  for (auto const& t : tracks) {
    const uint32_t node = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(TruthGraph::NodeRef{TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(t.trackId())});
    pdgId_.push_back(t.type());
    eventId_.push_back(packed);
    simTrackToVtx_.push_back(-1);
    trackIdToNode.emplace(t.trackId(), node);
  }

  // Production edge: track.vertIndex() is the local vector index into `vertices`.
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const int vi = tracks[i].vertIndex();
    if (vi < 0 || static_cast<std::size_t>(vi) >= vertices.size())
      continue;
    const uint32_t trkNode = baseVtx + static_cast<uint32_t>(vertices.size()) + static_cast<uint32_t>(i);
    const uint32_t prodVtxNode = baseVtx + static_cast<uint32_t>(vi);
    edges_.emplace_back(prodVtxNode, trkNode);
    edgeKinds_.push_back(static_cast<uint8_t>(TruthGraph::EdgeKind::Sim));
    simTrackToVtx_[trkNode] = static_cast<int32_t>(prodVtxNode);
  }

  // Decay edge: vertex.parentIndex() is the trackId of the parent track.
  for (auto const& v : vertices) {
    if (v.parentIndex() < 0)
      continue;
    auto it = trackIdToNode.find(static_cast<uint32_t>(v.parentIndex()));
    if (it == trackIdToNode.end())
      continue;
    auto vIt = vertexIdToNode.find(static_cast<uint32_t>(v.vertexId()));
    if (vIt == vertexIdToNode.end())
      continue;
    edges_.emplace_back(it->second, vIt->second);
    edgeKinds_.push_back(static_cast<uint8_t>(TruthGraph::EdgeKind::Sim));
  }
}

void TruthGraphAccumulator::accumulate(edm::Event const& event, edm::EventSetup const&) {
  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  event.getByLabel(simTrackTag_, tracks);
  event.getByLabel(simVertexTag_, vertices);
  if (tracks.isValid() && vertices.isValid())
    addSimSubEvent(*tracks, *vertices, EncodedEventId(0, 0));
}

void TruthGraphAccumulator::accumulate(PileUpEventPrincipal const& pep,
                                       edm::EventSetup const&,
                                       edm::StreamID const&) {
  const int bx = pep.bunchCrossing();
  if (!keepBx(bx))
    return;

  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  pep.getByLabel(simTrackTag_, tracks);
  pep.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;

  const int puIndex = ++pileupCountByBx_[bx];
  addSimSubEvent(*tracks, *vertices, EncodedEventId(bx, puIndex));
}

void TruthGraphAccumulator::finalizeEvent(edm::Event& event, edm::EventSetup const&) {
  auto out = std::make_unique<TruthGraph>();
  const uint32_t nNodes = static_cast<uint32_t>(nodes_.size());

  out->nodes = std::move(nodes_);
  out->pdgId = std::move(pdgId_);
  out->eventId = std::move(eventId_);
  out->simTrackToVtx = std::move(simTrackToVtx_);
  out->status.assign(nNodes, 0);
  out->statusFlags.assign(nNodes, 0);
  out->genEventOfNode.assign(nNodes, -1);
  out->simTrackToGen.assign(nNodes, -1);
  out->simVtxToGen.assign(nNodes, -1);

  // CSR out-edges (sorted by source).
  std::vector<uint32_t> order(edges_.size());
  for (uint32_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return edges_[a].first < edges_[b].first; });

  out->offsets.assign(nNodes + 1, 0);
  for (auto const& e : edges_)
    ++out->offsets[e.first + 1];
  for (uint32_t i = 1; i <= nNodes; ++i)
    out->offsets[i] += out->offsets[i - 1];

  out->edges.resize(edges_.size());
  out->edgeKind.resize(edges_.size());
  for (uint32_t k = 0; k < order.size(); ++k) {
    out->edges[k] = edges_[order[k]].second;
    out->edgeKind[k] = edgeKinds_[order[k]];
  }

  event.put(std::move(out));
}

DEFINE_DIGI_ACCUMULATOR(TruthGraphAccumulator);
