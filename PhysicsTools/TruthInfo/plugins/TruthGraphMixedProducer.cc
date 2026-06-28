// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Phase-A pileup prototype: build a raw TruthGraph from the MixingModule
// crossing frames (signal + pileup) instead of the signal-only g4SimHits.
//
// The MixCollection flattens all sub-events; SimTrack::trackId(),
// SimTrack::vertIndex(), SimVertex::vertexId() and SimVertex::parentIndex() are
// all local to each sub-event, so every linking map is keyed by
// (EncodedEventId, localId). Each node carries its EncodedEventId, so signal
// (bx=0,event=0) and pileup (bx!=0 or event!=0) sub-events stay distinguishable.
//
// This first prototype builds the SIM realm only (signal + pileup SimTracks /
// SimVertices). GEN merging is signal-only in the standard producer and is not
// reproduced here: pileup has no persisted GEN history anyway. Calo/tracker hit
// crossing frames are handled separately.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {
  // Mirror of TruthGraphProducer::packEventId so the logical producer / consumers
  // decode the bunch crossing the same way.
  uint64_t packEventId(EncodedEventId const& id) {
    uint64_t out = 0;
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }

  // Key for per-sub-event disambiguation: (EncodedEventId.rawId, local id).
  struct SubEventKey {
    uint32_t eid;
    uint32_t localId;
    bool operator==(SubEventKey const& o) const { return eid == o.eid && localId == o.localId; }
  };
  struct SubEventKeyHash {
    std::size_t operator()(SubEventKey const& k) const { return (static_cast<std::size_t>(k.eid) << 32) ^ k.localId; }
  };
}  // namespace

class TruthGraphMixedProducer : public edm::stream::EDProducer<> {
public:
  explicit TruthGraphMixedProducer(edm::ParameterSet const&);
  void produce(edm::Event&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<CrossingFrame<SimTrack>> simTrackToken_;
  const edm::EDGetTokenT<CrossingFrame<SimVertex>> simVertexToken_;
};

TruthGraphMixedProducer::TruthGraphMixedProducer(edm::ParameterSet const& cfg)
    : simTrackToken_(consumes<CrossingFrame<SimTrack>>(cfg.getParameter<edm::InputTag>("simTracks"))),
      simVertexToken_(consumes<CrossingFrame<SimVertex>>(cfg.getParameter<edm::InputTag>("simVertices"))) {
  produces<TruthGraph>();
}

void TruthGraphMixedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simTracks", edm::InputTag("mix", "g4SimHits"))
      ->setComment("CrossingFrame<SimTrack> from the MixingModule (signal + pileup).");
  desc.add<edm::InputTag>("simVertices", edm::InputTag("mix", "g4SimHits"))
      ->setComment("CrossingFrame<SimVertex> from the MixingModule (signal + pileup).");
  descriptions.addWithDefaultLabel(desc);
}

void TruthGraphMixedProducer::produce(edm::Event& evt, edm::EventSetup const&) {
  auto const& cfTracks = evt.get(simTrackToken_);
  auto const& cfVertices = evt.get(simVertexToken_);

  MixCollection<SimTrack> simTracks(&cfTracks);
  MixCollection<SimVertex> simVertices(&cfVertices);

  auto out = std::make_unique<TruthGraph>();

  const uint32_t nVtx = static_cast<uint32_t>(simVertices.size());
  const uint32_t nTrk = static_cast<uint32_t>(simTracks.size());
  const uint32_t nNodes = nVtx + nTrk;

  out->nodes().resize(nNodes);
  out->pdgId().assign(nNodes, 0);
  out->status().assign(nNodes, 0);
  out->statusFlags().assign(nNodes, 0);
  out->eventId().assign(nNodes, 0ull);
  out->genEventOfNode().assign(nNodes, -1);
  out->simTrackToGen().assign(nNodes, -1);
  out->simTrackToVtx().assign(nNodes, -1);
  out->simVtxToGen().assign(nNodes, -1);
  out->simVertexProcessType().assign(nNodes, 0);
  out->simTrackBackscattered().assign(nNodes, 0);

  // Vertex nodes come first (ids [0, nVtx)), tracks after (ids [nVtx, nNodes)).
  // perEventVtxNodes[eid] preserves sub-event order, so the position equals the
  // local vector index used by SimTrack::vertIndex().
  std::unordered_map<uint32_t, std::vector<uint32_t>> perEventVtxNodes;
  std::unordered_map<SubEventKey, uint32_t, SubEventKeyHash> trackKey;    // (eid, trackId) -> track node
  std::vector<std::pair<uint32_t, int>> vtxParent(nVtx, {0u, -1});        // (eid, parentTrackId) per vertex node
  std::vector<std::pair<uint32_t, int>> trkProdVtxLocal(nTrk, {0u, -1});  // (eid, local vertIndex) per track

  uint32_t v = 0;
  for (auto it = simVertices.begin(); it != simVertices.end(); ++it, ++v) {
    const uint32_t node = v;  // vertex nodes [0, nVtx)
    const EncodedEventId eid = it->eventId();
    out->nodes()[node] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(it->vertexId())};
    out->eventId()[node] = packEventId(eid);
    out->simVertexProcessType()[node] = static_cast<uint16_t>(it->processType());
    perEventVtxNodes[eid.rawId()].push_back(node);
    vtxParent[v] = {eid.rawId(), it->parentIndex()};
  }

  uint32_t t = 0;
  for (auto it = simTracks.begin(); it != simTracks.end(); ++it, ++t) {
    const uint32_t node = nVtx + t;  // track nodes [nVtx, nNodes)
    const EncodedEventId eid = it->eventId();
    out->nodes()[node] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(it->trackId())};
    out->pdgId()[node] = it->type();
    out->eventId()[node] = packEventId(eid);
    out->simTrackBackscattered()[node] = it->isFromBackScattering() ? 1 : 0;
    trackKey[SubEventKey{eid.rawId(), it->trackId()}] = node;
    trkProdVtxLocal[t] = {eid.rawId(), it->vertIndex()};
  }

  // Build the SIM bipartite edges, keyed within each sub-event.
  std::vector<std::pair<uint32_t, uint32_t>> edgePairs;
  std::vector<uint8_t> edgeKinds;
  edgePairs.reserve(2 * nTrk);
  edgeKinds.reserve(2 * nTrk);
  auto pushEdge = [&](uint32_t src, uint32_t dst) {
    edgePairs.emplace_back(src, dst);
    edgeKinds.emplace_back(static_cast<uint8_t>(TruthGraph::EdgeKind::Sim));
  };

  // Production edge: a track's production SimVertex is its sub-event-local vertIndex.
  for (uint32_t i = 0; i < nTrk; ++i) {
    const uint32_t trkNode = nVtx + i;
    const auto [eid, vi] = trkProdVtxLocal[i];
    if (vi < 0)
      continue;
    auto evIt = perEventVtxNodes.find(eid);
    if (evIt == perEventVtxNodes.end() || static_cast<std::size_t>(vi) >= evIt->second.size())
      continue;
    const uint32_t prodVtxNode = evIt->second[static_cast<std::size_t>(vi)];
    pushEdge(prodVtxNode, trkNode);  // vertex produces track
    out->simTrackToVtx()[trkNode] = static_cast<int32_t>(prodVtxNode);
  }

  // Decay edge: a vertex's parent track (SimVertex::parentIndex() is a trackId).
  for (uint32_t i = 0; i < nVtx; ++i) {
    const auto [eid, parentTrackId] = vtxParent[i];
    if (parentTrackId < 0)
      continue;
    auto kIt = trackKey.find(SubEventKey{eid, static_cast<uint32_t>(parentTrackId)});
    if (kIt == trackKey.end())
      continue;
    pushEdge(kIt->second, i);  // parent track decays into vertex
  }

  // CSR out-edges via the counting-sort cursor scatter: each edge lands in its
  // source's range, by construction (no sort, no permutation vector).
  out->offsets().assign(nNodes + 1, 0);
  for (auto const& e : edgePairs)
    ++out->offsets()[e.first + 1];
  for (uint32_t i = 1; i <= nNodes; ++i)
    out->offsets()[i] += out->offsets()[i - 1];

  out->edges().resize(edgePairs.size());
  out->edgeKind().resize(edgePairs.size());
  std::vector<uint32_t> cursor = out->offsets();
  for (std::size_t e = 0; e < edgePairs.size(); ++e) {
    const uint32_t pos = cursor[edgePairs[e].first]++;
    out->edges()[pos] = edgePairs[e].second;
    out->edgeKind()[pos] = edgeKinds[e];
  }

  if (!out->isConsistent())
    throw cms::Exception("TruthGraphMixedProducer") << "Produced TruthGraph is not consistent";

  evt.put(std::move(out));
}

DEFINE_FWK_MODULE(TruthGraphMixedProducer);
