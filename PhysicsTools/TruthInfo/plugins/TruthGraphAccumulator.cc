// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Phase-B (B1): build the mixed (signal + pileup) raw TruthGraph as a
// DigiAccumulatorMixMod, like TrackingTruthAccumulator / CaloTruthAccumulator.
// The framework hands us one sub-event at a time with its NATIVE
// SimTrack/SimVertex/HepMC collections, so trackId/vertIndex/parentIndex are used
// in their original local context (no flattening, no cross-pileup keying); the
// graph does not fragment the way the Phase-A MixCollection prototype did, it is
// identical for standard mixing and premixing, and it is consistent with the
// digis by construction.
//
// GEN handling is configurable per realm:
//   collapsePileupGen (default true) : for pileup, collapse the GEN decay chain to
//        the stable (status 1) GEN particles on a single gen vertex, keep the SIM
//        continuation (GenToSim links). This is the compact default the user asked
//        for; it also connects each pileup interaction into one component.
//   collapseSignalGen (default false): the signal keeps its full graph. Full GEN+SIM
//        for the signal reuses the standard TruthGraphProducer build and is staged;
//        until then collapseSignalGen=false leaves the signal as SIM-only here.
//
//   pileupBunchCrossings (default {0} = in-time pileup only): which bunch crossings
//        to include for pileup.
//
// Each node carries an EncodedEventId: (0,0) for the signal, (bunchCrossing,
// pileupIndex) for pileup, so signal and pileup stay distinguishable.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"

#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {
  uint64_t packEventId(EncodedEventId const& id) {
    uint64_t out = 0;
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }

  // Stable (status 1) GEN particles as (barcode, pdgId). Used to collapse the GEN
  // part to "stable particles on a single gen vertex".
  std::vector<std::pair<int, int>> stableFromHepMC2(HepMC::GenEvent const& ev) {
    std::vector<std::pair<int, int>> out;
    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      if (*p != nullptr && (*p)->status() == 1)
        out.emplace_back((*p)->barcode(), (*p)->pdg_id());
    }
    return out;
  }

  std::vector<std::pair<int, int>> stableFromHepMC3(HepMC3::GenEvent const& ev) {
    std::vector<std::pair<int, int>> out;
    for (auto const& p : ev.particles()) {
      if (p && p->status() == 1)
        out.emplace_back(p->id(), p->pid());
    }
    return out;
  }

  // Read the stable GEN particles from a signal Event or a PileUpEventPrincipal
  // (both expose getByLabel), preferring HepMC3.
  template <class EvT>
  std::vector<std::pair<int, int>> readStableGen(EvT const& ev,
                                                 edm::InputTag const& hepmc3Tag,
                                                 edm::InputTag const& hepmc2Tag) {
    edm::Handle<edm::HepMC3Product> h3;
    if (ev.getByLabel(hepmc3Tag, h3) && h3.isValid() && h3->GetEvent() != nullptr) {
      HepMC3::GenEvent ev3;
      ev3.read_data(*h3->GetEvent());
      return stableFromHepMC3(ev3);
    }
    edm::Handle<edm::HepMCProduct> h2;
    if (ev.getByLabel(hepmc2Tag, h2) && h2.isValid() && h2->GetEvent() != nullptr)
      return stableFromHepMC2(*h2->GetEvent());
    return {};
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
  // Append one sub-event. SimTrack/SimVertex ids are local to this sub-event. If
  // `stableGen` is non-empty, also add a single collapsed gen vertex with those
  // stable particles and GenToSim links to the primary SimTracks.
  void addSubEvent(std::vector<std::pair<int, int>> const& stableGen,
                   edm::SimTrackContainer const& tracks,
                   edm::SimVertexContainer const& vertices,
                   EncodedEventId const& eid);

  const edm::InputTag simTrackTag_;
  const edm::InputTag simVertexTag_;
  const edm::InputTag hepmc3Tag_;
  const edm::InputTag hepmc2Tag_;
  const std::vector<int> pileupBunchCrossings_;
  const bool collapsePileupGen_;
  const bool collapseSignalGen_;

  std::unordered_map<int, int> pileupCountByBx_;

  std::vector<TruthGraph::NodeRef> nodes_;
  std::vector<int32_t> pdgId_;
  std::vector<int16_t> status_;
  std::vector<uint64_t> eventId_;
  std::vector<int32_t> simTrackToVtx_;
  std::vector<int32_t> simTrackToGen_;
  std::vector<std::pair<uint32_t, uint32_t>> edges_;
  std::vector<uint8_t> edgeKinds_;
  std::vector<uint16_t> simVertexProcessType_;  // node-parallel; G4 process subtype (SimVertex only)
  std::vector<uint8_t> simTrackBackscattered_;  // node-parallel; albedo flag (SimTrack only)

  [[nodiscard]] bool keepBx(int bx) const {
    return std::find(pileupBunchCrossings_.begin(), pileupBunchCrossings_.end(), bx) != pileupBunchCrossings_.end();
  }
};

TruthGraphAccumulator::TruthGraphAccumulator(edm::ParameterSet const& cfg,
                                             edm::ProducesCollector producesCollector,
                                             edm::ConsumesCollector& iC)
    : simTrackTag_(cfg.getParameter<edm::InputTag>("simTracks")),
      simVertexTag_(cfg.getParameter<edm::InputTag>("simVertices")),
      hepmc3Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC3")),
      hepmc2Tag_(cfg.getParameter<edm::InputTag>("genEventHepMC")),
      pileupBunchCrossings_(cfg.getParameter<std::vector<int>>("pileupBunchCrossings")),
      collapsePileupGen_(cfg.getParameter<bool>("collapsePileupGen")),
      collapseSignalGen_(cfg.getParameter<bool>("collapseSignalGen")) {
  producesCollector.produces<TruthGraph>();
  iC.consumes<edm::SimTrackContainer>(simTrackTag_);
  iC.consumes<edm::SimVertexContainer>(simVertexTag_);
  iC.mayConsume<edm::HepMC3Product>(hepmc3Tag_);
  iC.mayConsume<edm::HepMCProduct>(hepmc2Tag_);
}

void TruthGraphAccumulator::initializeEvent(edm::Event const&, edm::EventSetup const&) {
  pileupCountByBx_.clear();
  nodes_.clear();
  pdgId_.clear();
  status_.clear();
  eventId_.clear();
  simTrackToVtx_.clear();
  simTrackToGen_.clear();
  edges_.clear();
  edgeKinds_.clear();
  simVertexProcessType_.clear();
  simTrackBackscattered_.clear();
}

void TruthGraphAccumulator::addSubEvent(std::vector<std::pair<int, int>> const& stableGen,
                                        edm::SimTrackContainer const& tracks,
                                        edm::SimVertexContainer const& vertices,
                                        EncodedEventId const& eid) {
  const uint64_t packed = packEventId(eid);
  auto pushNode = [&](TruthGraph::NodeKind kind, int64_t key, int32_t pdg, int16_t st) {
    const uint32_t node = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(TruthGraph::NodeRef{kind, key});
    pdgId_.push_back(pdg);
    status_.push_back(st);
    eventId_.push_back(packed);
    simTrackToVtx_.push_back(-1);
    simTrackToGen_.push_back(-1);
    simVertexProcessType_.push_back(0);
    simTrackBackscattered_.push_back(0);
    return node;
  };
  auto pushEdge = [&](uint32_t src, uint32_t dst, TruthGraph::EdgeKind k) {
    edges_.emplace_back(src, dst);
    edgeKinds_.push_back(static_cast<uint8_t>(k));
  };

  // Collapsed GEN: one gen vertex + the stable gen particles. barcode -> node.
  std::unordered_map<int, uint32_t> genBarcodeToNode;
  int32_t genVtxNode = -1;
  if (!stableGen.empty()) {
    genVtxNode = static_cast<int32_t>(pushNode(TruthGraph::NodeKind::GenVertex, 0, 0, 0));
    genBarcodeToNode.reserve(stableGen.size() * 2);
    for (auto const& [barcode, pdg] : stableGen) {
      const uint32_t pn = pushNode(TruthGraph::NodeKind::GenParticle, barcode, pdg, 1);
      pushEdge(static_cast<uint32_t>(genVtxNode), pn, TruthGraph::EdgeKind::Gen);
      genBarcodeToNode.emplace(barcode, pn);
    }
  }

  // SIM realm (native local ids).
  std::unordered_map<uint32_t, uint32_t> vertexIdToNode;
  vertexIdToNode.reserve(vertices.size() * 2);
  const uint32_t baseVtx = static_cast<uint32_t>(nodes_.size());
  for (auto const& v : vertices) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(v.vertexId()), 0, 0);
    simVertexProcessType_[node] = static_cast<uint16_t>(v.processType());
    vertexIdToNode.emplace(static_cast<uint32_t>(v.vertexId()), node);
  }
  const uint32_t baseTrk = static_cast<uint32_t>(nodes_.size());
  std::unordered_map<uint32_t, uint32_t> trackIdToNode;
  trackIdToNode.reserve(tracks.size() * 2);
  for (auto const& t : tracks) {
    const uint32_t node = pushNode(TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(t.trackId()), t.type(), 0);
    simTrackBackscattered_[node] = t.isFromBackScattering() ? 1 : 0;
    trackIdToNode.emplace(t.trackId(), node);
  }

  // Production edge: track.vertIndex() is the local vector index into `vertices`.
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const int vi = tracks[i].vertIndex();
    if (vi < 0 || static_cast<std::size_t>(vi) >= vertices.size())
      continue;
    const uint32_t trkNode = baseTrk + static_cast<uint32_t>(i);
    const uint32_t prodVtxNode = baseVtx + static_cast<uint32_t>(vi);
    pushEdge(prodVtxNode, trkNode, TruthGraph::EdgeKind::Sim);
    simTrackToVtx_[trkNode] = static_cast<int32_t>(prodVtxNode);
  }

  // Decay edge: vertex.parentIndex() is the trackId of the parent track.
  for (auto const& v : vertices) {
    if (v.parentIndex() < 0)
      continue;
    auto pIt = trackIdToNode.find(static_cast<uint32_t>(v.parentIndex()));
    auto vIt = vertexIdToNode.find(static_cast<uint32_t>(v.vertexId()));
    if (pIt != trackIdToNode.end() && vIt != vertexIdToNode.end())
      pushEdge(pIt->second, vIt->second, TruthGraph::EdgeKind::Sim);
  }

  // GenToSim: a primary SimTrack's genpartIndex is the stable particle's barcode.
  if (!genBarcodeToNode.empty()) {
    for (auto const& t : tracks) {
      auto gIt = genBarcodeToNode.find(t.genpartIndex());
      if (gIt == genBarcodeToNode.end())
        continue;
      auto sIt = trackIdToNode.find(t.trackId());
      if (sIt == trackIdToNode.end())
        continue;
      pushEdge(gIt->second, sIt->second, TruthGraph::EdgeKind::GenToSim);
      simTrackToGen_[sIt->second] = static_cast<int32_t>(gIt->second);
    }
  }
}

void TruthGraphAccumulator::accumulate(edm::Event const& event, edm::EventSetup const&) {
  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  event.getByLabel(simTrackTag_, tracks);
  event.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;
  std::vector<std::pair<int, int>> stableGen;
  if (collapseSignalGen_)
    stableGen = readStableGen(event, hepmc3Tag_, hepmc2Tag_);
  addSubEvent(stableGen, *tracks, *vertices, EncodedEventId(0, 0));
}

void TruthGraphAccumulator::accumulate(PileUpEventPrincipal const& pep, edm::EventSetup const&, edm::StreamID const&) {
  const int bx = pep.bunchCrossing();
  if (!keepBx(bx))
    return;

  edm::Handle<edm::SimTrackContainer> tracks;
  edm::Handle<edm::SimVertexContainer> vertices;
  pep.getByLabel(simTrackTag_, tracks);
  pep.getByLabel(simVertexTag_, vertices);
  if (!tracks.isValid() || !vertices.isValid())
    return;

  std::vector<std::pair<int, int>> stableGen;
  if (collapsePileupGen_)
    stableGen = readStableGen(pep, hepmc3Tag_, hepmc2Tag_);

  const int puIndex = ++pileupCountByBx_[bx];
  addSubEvent(stableGen, *tracks, *vertices, EncodedEventId(bx, puIndex));
}

void TruthGraphAccumulator::finalizeEvent(edm::Event& event, edm::EventSetup const&) {
  auto out = std::make_unique<TruthGraph>();
  const uint32_t nNodes = static_cast<uint32_t>(nodes_.size());

  out->nodes = std::move(nodes_);
  out->pdgId = std::move(pdgId_);
  out->status = std::move(status_);
  out->eventId = std::move(eventId_);
  out->simTrackToVtx = std::move(simTrackToVtx_);
  out->simTrackToGen = std::move(simTrackToGen_);
  out->simVertexProcessType = std::move(simVertexProcessType_);
  out->simTrackBackscattered = std::move(simTrackBackscattered_);
  out->statusFlags.assign(nNodes, 0);
  out->genEventOfNode.assign(nNodes, -1);
  out->simVtxToGen.assign(nNodes, -1);

  // CSR out-edges via the counting-sort cursor scatter: each edge lands in its
  // source's range, by construction (no sort, no permutation vector).
  out->offsets.assign(nNodes + 1, 0);
  for (auto const& e : edges_)
    ++out->offsets[e.first + 1];
  for (uint32_t i = 1; i <= nNodes; ++i)
    out->offsets[i] += out->offsets[i - 1];

  out->edges.resize(edges_.size());
  out->edgeKind.resize(edges_.size());
  std::vector<uint32_t> cursor = out->offsets;
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const uint32_t pos = cursor[edges_[e].first]++;
    out->edges[pos] = edges_[e].second;
    out->edgeKind[pos] = edgeKinds_[e];
  }

  if (!out->isConsistent())
    throw cms::Exception("TruthGraphAccumulator") << "Produced TruthGraph is not consistent";

  event.put(std::move(out));
}

DEFINE_DIGI_ACCUMULATOR(TruthGraphAccumulator);
