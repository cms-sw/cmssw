// Author: Felice Pantaleo - CERN
// Date: 03/2026
//
// Build a logical truth::Graph from the raw heterogeneous TruthGraph.
// The topology comes from the raw TruthGraph.
// Standalone payload (momentum/position/checkpoints) is materialized from optional
// HepMC2 / HepMC3 / SimTrack / SimVertex inputs.
//
// GenParticle and SimTrack nodes are merged when a robust association exists.
// The corresponding production GenVertex and SimVertex nodes are also merged,
// but only for locally one-to-one GenVertex <-> SimVertex associations induced
// by the same GenParticle <-> SimTrack match. This avoids creating artificial
// high-degree merged vertices from transitive many-to-one vertex matches.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// Legacy HepMC (HepMC2)
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

// HepMC3
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {

  struct DSU {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit DSU(int n) : parent(n), rank(n, 0) {
      for (int i = 0; i < n; ++i)
        parent[i] = i;
    }

    int find(int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    }

    void unite(int a, int b) {
      a = find(a);
      b = find(b);
      if (a == b)
        return;
      if (rank[a] < rank[b])
        std::swap(a, b);
      parent[b] = a;
      if (rank[a] == rank[b])
        ++rank[a];
    }
  };

  struct GenParticlePayload {
    int32_t pdgId = 0;
    int16_t status = 0;
    math::XYZTLorentzVectorD momentum;
  };

  struct GenVertexPayload {
    math::XYZTLorentzVectorD position;
  };

  struct GraphPostProcessingConfig {
    bool collapseIntermediateGenParticles = true;
  };

  bool isParticleKind(TruthGraph::NodeKind kind) {
    return kind == TruthGraph::NodeKind::GenParticle || kind == TruthGraph::NodeKind::SimTrack;
  }

  bool isVertexKind(TruthGraph::NodeKind kind) {
    return kind == TruthGraph::NodeKind::GenVertex || kind == TruthGraph::NodeKind::SimVertex;
  }

  bool isGenParticleToSimTrack(TruthGraph const& g, uint32_t src, uint32_t dst) {
    auto const& s = g.nodeRef(src);
    auto const& d = g.nodeRef(dst);
    return s.kind == TruthGraph::NodeKind::GenParticle && d.kind == TruthGraph::NodeKind::SimTrack;
  }

  void buildCSR(uint32_t nSources,
                std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                std::vector<uint32_t>& offsets,
                std::vector<uint32_t>& flat) {
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    offsets.assign(nSources + 1, 0);
    for (auto const& e : pairs) {
      ++offsets[e.first + 1];
    }
    for (uint32_t i = 1; i <= nSources; ++i) {
      offsets[i] += offsets[i - 1];
    }

    flat.assign(pairs.size(), 0);
    auto cursor = offsets;
    for (auto const& e : pairs) {
      flat[cursor[e.first]++] = e.second;
    }
  }

  template <typename HandleT>
  bool validHandle(HandleT const& h) {
    return h.isValid();
  }

  void fillGenPayloadFromHepMC2(HepMC::GenEvent const& ev,
                                std::unordered_map<int, GenParticlePayload>& particlePayload,
                                std::unordered_map<int, GenVertexPayload>& vertexPayload) {
    particlePayload.reserve(ev.particles_size() * 2);
    vertexPayload.reserve(ev.vertices_size() * 2);

    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      if (*p == nullptr)
        continue;

      const int bc = (*p)->barcode();

      GenParticlePayload payload;
      payload.pdgId = (*p)->pdg_id();
      payload.status = static_cast<int16_t>((*p)->status());
      payload.momentum = math::XYZTLorentzVectorD(
          (*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());

      particlePayload.emplace(bc, payload);
    }

    for (auto v = ev.vertices_begin(); v != ev.vertices_end(); ++v) {
      if (*v == nullptr)
        continue;

      const int bc = (*v)->barcode();

      GenVertexPayload payload;
      payload.position = math::XYZTLorentzVectorD(
          (*v)->position().x(), (*v)->position().y(), (*v)->position().z(), (*v)->position().t());

      vertexPayload.emplace(bc, payload);
    }
  }

  void fillGenPayloadFromHepMC3(HepMC3::GenEvent const& ev,
                                std::unordered_map<int, GenParticlePayload>& particlePayload,
                                std::unordered_map<int, GenVertexPayload>& vertexPayload) {
    particlePayload.reserve(ev.particles().size() * 2);
    vertexPayload.reserve(ev.vertices().size() * 2);

    for (auto const& pptr : ev.particles()) {
      if (!pptr)
        continue;

      const int bc = pptr->id();

      GenParticlePayload payload;
      payload.pdgId = pptr->pid();
      payload.status = static_cast<int16_t>(pptr->status());
      payload.momentum = math::XYZTLorentzVectorD(
          pptr->momentum().px(), pptr->momentum().py(), pptr->momentum().pz(), pptr->momentum().e());

      particlePayload.emplace(bc, payload);
    }

    for (auto const& vptr : ev.vertices()) {
      if (!vptr)
        continue;

      const int bc = vptr->id();

      GenVertexPayload payload;
      payload.position = math::XYZTLorentzVectorD(
          vptr->position().x(), vptr->position().y(), vptr->position().z(), vptr->position().t());

      vertexPayload.emplace(bc, payload);
    }
  }

  int32_t genParticlePdgId(TruthGraph const& raw,
                           uint32_t nodeId,
                           std::unordered_map<int, GenParticlePayload> const& genParticlePayload) {
    auto const& ref = raw.nodeRef(nodeId);

    if (ref.kind != TruthGraph::NodeKind::GenParticle)
      return 0;

    const auto rawPdgId = raw.nodePdgId(nodeId);
    if (rawPdgId != 0)
      return rawPdgId;

    const int barcode = static_cast<int>(ref.key);
    auto it = genParticlePayload.find(barcode);
    if (it != genParticlePayload.end())
      return it->second.pdgId;

    return 0;
  }

  std::vector<uint8_t> buildKeepMaskForMotherPdgId(TruthGraph const& raw,
                                                   std::unordered_map<int, GenParticlePayload> const& genParticlePayload,
                                                   int32_t motherPdgId) {
    const uint32_t nRawNodes = raw.nNodes();

    std::vector<uint8_t> keep(nRawNodes, 1);

    if (motherPdgId == 0)
      return keep;

    keep.assign(nRawNodes, 0);

    std::queue<uint32_t> queue;

    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      auto const& ref = raw.nodeRef(nodeId);

      if (ref.kind != TruthGraph::NodeKind::GenParticle)
        continue;

      if (genParticlePdgId(raw, nodeId, genParticlePayload) != motherPdgId)
        continue;

      if (!keep[nodeId]) {
        keep[nodeId] = 1;
        queue.push(nodeId);
      }
    }

    while (!queue.empty()) {
      const uint32_t src = queue.front();
      queue.pop();

      for (uint32_t dst : raw.children(src)) {
        if (dst >= nRawNodes)
          continue;

        if (!keep[dst]) {
          keep[dst] = 1;
          queue.push(dst);
        }
      }
    }

    return keep;
  }

  std::vector<int32_t> buildGenParticleToProductionGenVertexMap(TruthGraph const& raw,
                                                                std::vector<uint8_t> const& keepRawNode) {
    const uint32_t nRawNodes = raw.nNodes();
    std::vector<int32_t> genParticleToProductionGenVertex(nRawNodes, -1);

    for (uint32_t src = 0; src < nRawNodes; ++src) {
      if (!keepRawNode[src])
        continue;

      if (raw.nodeRef(src).kind != TruthGraph::NodeKind::GenVertex)
        continue;

      const auto dsts = raw.children(src);
      const auto edgeKinds = raw.childrenEdgeKinds(src);

      for (std::size_t i = 0; i < dsts.size(); ++i) {
        const uint32_t dst = dsts[i];

        if (dst >= nRawNodes || !keepRawNode[dst])
          continue;

        if (raw.nodeRef(dst).kind != TruthGraph::NodeKind::GenParticle)
          continue;

        if (static_cast<TruthGraph::EdgeKind>(edgeKinds[i]) != TruthGraph::EdgeKind::Gen)
          continue;

        if (genParticleToProductionGenVertex[dst] < 0) {
          genParticleToProductionGenVertex[dst] = static_cast<int32_t>(src);
        }
      }
    }

    return genParticleToProductionGenVertex;
  }

  void buildRawSimVertexDegrees(TruthGraph const& raw,
                                std::vector<uint8_t> const& keepRawNode,
                                std::vector<uint32_t>& simVertexIncomingSimTracks,
                                std::vector<uint32_t>& simVertexOutgoingSimTracks) {
    const uint32_t nRawNodes = raw.nNodes();

    simVertexIncomingSimTracks.assign(nRawNodes, 0);
    simVertexOutgoingSimTracks.assign(nRawNodes, 0);

    for (uint32_t src = 0; src < nRawNodes; ++src) {
      if (!keepRawNode[src])
        continue;

      const auto dsts = raw.children(src);
      const auto edgeKinds = raw.childrenEdgeKinds(src);

      for (std::size_t i = 0; i < dsts.size(); ++i) {
        const uint32_t dst = dsts[i];

        if (dst >= nRawNodes || !keepRawNode[dst])
          continue;

        if (static_cast<TruthGraph::EdgeKind>(edgeKinds[i]) != TruthGraph::EdgeKind::Sim)
          continue;

        const auto srcKind = raw.nodeRef(src).kind;
        const auto dstKind = raw.nodeRef(dst).kind;

        if (srcKind == TruthGraph::NodeKind::SimTrack && dstKind == TruthGraph::NodeKind::SimVertex) {
          ++simVertexIncomingSimTracks[dst];
        } else if (srcKind == TruthGraph::NodeKind::SimVertex && dstKind == TruthGraph::NodeKind::SimTrack) {
          ++simVertexOutgoingSimTracks[src];
        }
      }
    }
  }

  bool directCollapsibleGenParticleChain(truth::Graph const& graph,
                                         uint32_t particleId,
                                         uint32_t& childId,
                                         uint32_t& decayVertexId) {
    if (particleId >= graph.nParticles())
      return false;

    auto const& particle = graph.particles[particleId];

    if (!particle.hasGen())
      return false;

    if (particle.status == 1)
      return false;

    if (particle.pdgId == 0)
      return false;

    const auto decayVertices = graph.decayVertices(particleId);
    if (decayVertices.size() != 1)
      return false;

    const uint32_t vertexId = decayVertices.front();
    if (vertexId >= graph.nVertices())
      return false;

    auto const& vertex = graph.vertices[vertexId];

    if (!vertex.hasGen() || vertex.hasSim())
      return false;

    const auto incoming = graph.incomingParticles(vertexId);
    const auto outgoing = graph.outgoingParticles(vertexId);

    if (incoming.size() != 1 || incoming.front() != particleId)
      return false;

    if (outgoing.size() != 1)
      return false;

    const uint32_t candidateChild = outgoing.front();
    if (candidateChild >= graph.nParticles())
      return false;

    if (candidateChild == particleId)
      return false;

    auto const& child = graph.particles[candidateChild];

    if (!child.hasGen())
      return false;

    if (child.pdgId != particle.pdgId)
      return false;

    childId = candidateChild;
    decayVertexId = vertexId;
    return true;
  }

  truth::Graph collapseIntermediateGenParticleChains(truth::Graph const& input) {
    if (input.empty())
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    std::vector<int32_t> directChild(nParticles, -1);
    std::vector<uint8_t> skipVertex(nVertices, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      uint32_t childId = 0;
      uint32_t decayVertexId = 0;

      if (!directCollapsibleGenParticleChain(input, particleId, childId, decayVertexId))
        continue;

      directChild[particleId] = static_cast<int32_t>(childId);
      skipVertex[decayVertexId] = 1;
    }

    std::vector<uint32_t> particleRepresentative(nParticles, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      uint32_t current = particleId;

      for (uint32_t step = 0; step < nParticles; ++step) {
        const int32_t next = directChild[current];
        if (next < 0)
          break;

        current = static_cast<uint32_t>(next);
      }

      particleRepresentative[particleId] = current;
    }

    truth::Graph output;

    std::unordered_map<uint32_t, uint32_t> representativeToNewParticle;
    representativeToNewParticle.reserve(nParticles);

    std::vector<int32_t> oldParticleToNew(nParticles, -1);

    for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
      const uint32_t representative = particleRepresentative[oldParticle];

      auto inserted =
          representativeToNewParticle.emplace(representative, static_cast<uint32_t>(output.particles.size()));
      const uint32_t newParticle = inserted.first->second;

      if (inserted.second) {
        output.particles.push_back(input.particles[representative]);
      }

      oldParticleToNew[oldParticle] = static_cast<int32_t>(newParticle);
    }

    std::vector<uint8_t> keepVertex(nVertices, 0);

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      if (skipVertex[oldVertex])
        continue;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle < oldParticleToNew.size() && oldParticleToNew[oldParticle] >= 0) {
          keepVertex[oldVertex] = 1;
          break;
        }
      }

      if (keepVertex[oldVertex])
        continue;

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle < oldParticleToNew.size() && oldParticleToNew[oldParticle] >= 0) {
          keepVertex[oldVertex] = 1;
          break;
        }
      }
    }

    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      if (!keepVertex[oldVertex])
        continue;

      oldVertexToNew[oldVertex] = static_cast<int32_t>(output.vertices.size());
      output.vertices.push_back(input.vertices[oldVertex]);
    }

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const int32_t newVertex = oldVertexToNew[oldVertex];
      if (newVertex < 0)
        continue;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle >= oldParticleToNew.size())
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), static_cast<uint32_t>(newVertex));
        vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
      }

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle >= oldParticleToNew.size())
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        vertexToOutgoingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
        particleToProductionVertexPairs.emplace_back(static_cast<uint32_t>(newParticle),
                                                     static_cast<uint32_t>(newVertex));
      }
    }

    buildCSR(output.nParticles(),
             particleToDecayVertexPairs,
             output.particleToDecayVertexOffsets,
             output.particleToDecayVertices);

    buildCSR(output.nParticles(),
             particleToProductionVertexPairs,
             output.particleToProductionVertexOffsets,
             output.particleToProductionVertices);

    buildCSR(output.nVertices(),
             vertexToOutgoingParticlePairs,
             output.vertexToOutgoingParticleOffsets,
             output.vertexToOutgoingParticles);

    buildCSR(output.nVertices(),
             vertexToIncomingParticlePairs,
             output.vertexToIncomingParticleOffsets,
             output.vertexToIncomingParticles);

    return output;
  }

  truth::Graph postProcessGraph(truth::Graph input, GraphPostProcessingConfig const& config) {
    if (config.collapseIntermediateGenParticles) {
      input = collapseIntermediateGenParticleChains(input);
    }

    return input;
  }

}  // namespace

class TruthLogicalGraphProducer : public edm::stream::EDProducer<> {
public:
  explicit TruthLogicalGraphProducer(edm::ParameterSet const& cfg)
      : srcToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("src"))),
        simTrackToken_(mayConsume<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
        simVertexToken_(mayConsume<edm::SimVertexContainer>(cfg.getParameter<edm::InputTag>("simVertices"))),
        hepmc3Token_(mayConsume<edm::HepMC3Product>(cfg.getParameter<edm::InputTag>("genEventHepMC3"))),
        hepmc2Token_(mayConsume<edm::HepMCProduct>(cfg.getParameter<edm::InputTag>("genEventHepMC"))),
        motherPdgId_(cfg.getParameter<int32_t>("motherPdgId")),
        mergeGenSimVertices_(cfg.getParameter<bool>("mergeGenSimVertices")) {
    postProcessing_.collapseIntermediateGenParticles = cfg.getParameter<bool>("collapseIntermediateGenParticles");
    produces<truth::Graph>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("truthGraphProducer"));
    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
    desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
    desc.add<edm::InputTag>("genEventHepMC3", edm::InputTag("generatorSmeared"));
    desc.add<edm::InputTag>("genEventHepMC", edm::InputTag("generatorSmeared"));
    desc.add<int32_t>("motherPdgId", 0)
        ->setComment("If non-zero, keep only GEN particles with this exact PDG id and all their descendants.");
    desc.add<bool>("mergeGenSimVertices", true)
        ->setComment(
            "If true, merge production GenVertex and SimVertex only for locally one-to-one matches induced by "
            "GenParticle <-> SimTrack associations.");
    desc.add<bool>("collapseIntermediateGenParticles", true)
        ->setComment(
            "If true, collapse GEN chains P -> V -> C where P has status != 1, C is the only daughter, "
            "and P and C have the same PDG id.");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::Event& evt, edm::EventSetup const&) override {
    auto const& raw = evt.get(srcToken_);

    if (!raw.isConsistent()) {
      throw cms::Exception("TruthLogicalGraphProducer") << "Input TruthGraph is not consistent";
    }

    auto out = std::make_unique<truth::Graph>();

    const uint32_t nRawNodes = raw.nNodes();

    edm::Handle<edm::SimTrackContainer> hSimTracks;
    evt.getByToken(simTrackToken_, hSimTracks);

    edm::Handle<edm::SimVertexContainer> hSimVertices;
    evt.getByToken(simVertexToken_, hSimVertices);

    std::unordered_map<uint32_t, uint32_t> simTrackIdToIndex;
    if (validHandle(hSimTracks)) {
      simTrackIdToIndex.reserve(hSimTracks->size() * 2);
      for (uint32_t i = 0; i < hSimTracks->size(); ++i) {
        simTrackIdToIndex.emplace((*hSimTracks)[i].trackId(), i);
      }
    }

    std::unordered_map<int, GenParticlePayload> genParticlePayload;
    std::unordered_map<int, GenVertexPayload> genVertexPayload;

    bool haveGenPayload = false;

    {
      edm::Handle<edm::HepMC3Product> h3;
      evt.getByToken(hepmc3Token_, h3);
      if (validHandle(h3) && h3->GetEvent() != nullptr) {
        const HepMC3::GenEventData* data = h3->GetEvent();
        HepMC3::GenEvent ev3;
        ev3.read_data(*data);
        fillGenPayloadFromHepMC3(ev3, genParticlePayload, genVertexPayload);
        haveGenPayload = true;
      }
    }

    if (!haveGenPayload) {
      edm::Handle<edm::HepMCProduct> h2;
      evt.getByToken(hepmc2Token_, h2);
      if (validHandle(h2) && h2->GetEvent() != nullptr) {
        fillGenPayloadFromHepMC2(*h2->GetEvent(), genParticlePayload, genVertexPayload);
        haveGenPayload = true;
      }
    }

    const auto keepRawNode = buildKeepMaskForMotherPdgId(raw, genParticlePayload, motherPdgId_);
    const auto genParticleToProductionGenVertex = buildGenParticleToProductionGenVertexMap(raw, keepRawNode);

    std::vector<uint32_t> rawSimVertexIncomingSimTracks;
    std::vector<uint32_t> rawSimVertexOutgoingSimTracks;
    buildRawSimVertexDegrees(raw, keepRawNode, rawSimVertexIncomingSimTracks, rawSimVertexOutgoingSimTracks);

    // ------------------------------------------------------------------
    // 1. Temporary ids
    // ------------------------------------------------------------------
    std::vector<int32_t> rawToParticleTmp(nRawNodes, -1);
    std::vector<int32_t> rawToVertexTmp(nRawNodes, -1);

    int nParticleTmp = 0;
    int nVertexTmp = 0;

    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      if (!keepRawNode[nodeId])
        continue;

      const auto kind = raw.nodeRef(nodeId).kind;
      if (isParticleKind(kind)) {
        rawToParticleTmp[nodeId] = nParticleTmp++;
      } else if (isVertexKind(kind)) {
        rawToVertexTmp[nodeId] = nVertexTmp++;
      }
    }

    DSU particleDSU(nParticleTmp);
    DSU vertexDSU(nVertexTmp);

    std::vector<std::pair<uint32_t, uint32_t>> productionVertexMergeCandidates;

    auto addProductionVertexMergeCandidate = [&](uint32_t simTrackNode, uint32_t genParticleNode) {
      if (!mergeGenSimVertices_)
        return;

      const int32_t simVertexNode = raw.nodeSimTrackToVtx(simTrackNode);
      if (simVertexNode < 0)
        return;

      if (genParticleNode >= genParticleToProductionGenVertex.size())
        return;

      const int32_t genVertexNode = genParticleToProductionGenVertex[genParticleNode];
      if (genVertexNode < 0)
        return;

      const uint32_t gv = static_cast<uint32_t>(genVertexNode);
      const uint32_t sv = static_cast<uint32_t>(simVertexNode);

      if (gv >= nRawNodes || sv >= nRawNodes)
        return;

      if (!keepRawNode[gv] || !keepRawNode[sv])
        return;

      if (rawToVertexTmp[gv] < 0 || rawToVertexTmp[sv] < 0)
        return;

      if (raw.nodeRef(gv).kind != TruthGraph::NodeKind::GenVertex)
        return;

      if (raw.nodeRef(sv).kind != TruthGraph::NodeKind::SimVertex)
        return;

      productionVertexMergeCandidates.emplace_back(gv, sv);
    };

    // ------------------------------------------------------------------
    // 2. Merge particles across GEN <-> SIM and collect vertex merge candidates
    // ------------------------------------------------------------------
    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      if (!keepRawNode[nodeId])
        continue;

      auto const& ref = raw.nodeRef(nodeId);
      if (ref.kind != TruthGraph::NodeKind::SimTrack)
        continue;

      const int32_t genNode = raw.nodeSimTrackToGen(nodeId);
      if (genNode < 0)
        continue;

      const uint32_t genNodeU32 = static_cast<uint32_t>(genNode);
      if (genNodeU32 >= nRawNodes)
        continue;

      if (!keepRawNode[genNodeU32])
        continue;

      if (raw.nodeRef(genNodeU32).kind != TruthGraph::NodeKind::GenParticle)
        continue;

      if (rawToParticleTmp[nodeId] < 0 || rawToParticleTmp[genNodeU32] < 0)
        continue;

      particleDSU.unite(rawToParticleTmp[nodeId], rawToParticleTmp[genNodeU32]);
      addProductionVertexMergeCandidate(nodeId, genNodeU32);
    }

    for (uint32_t src = 0; src < nRawNodes; ++src) {
      if (!keepRawNode[src])
        continue;

      const auto dsts = raw.children(src);
      const auto ekinds = raw.childrenEdgeKinds(src);

      for (std::size_t i = 0; i < dsts.size(); ++i) {
        const uint32_t dst = dsts[i];

        if (dst >= nRawNodes || !keepRawNode[dst])
          continue;

        const auto ek = static_cast<TruthGraph::EdgeKind>(ekinds[i]);

        if (ek != TruthGraph::EdgeKind::GenToSim && ek != TruthGraph::EdgeKind::SimToGen)
          continue;

        if (!isGenParticleToSimTrack(raw, src, dst))
          continue;

        if (rawToParticleTmp[src] < 0 || rawToParticleTmp[dst] < 0)
          continue;

        particleDSU.unite(rawToParticleTmp[src], rawToParticleTmp[dst]);
        addProductionVertexMergeCandidate(dst, src);
      }
    }

    if (mergeGenSimVertices_) {
      std::sort(productionVertexMergeCandidates.begin(), productionVertexMergeCandidates.end());
      productionVertexMergeCandidates.erase(
          std::unique(productionVertexMergeCandidates.begin(), productionVertexMergeCandidates.end()),
          productionVertexMergeCandidates.end());

      std::vector<uint16_t> genVertexCandidateMultiplicity(nRawNodes, 0);
      std::vector<uint16_t> simVertexCandidateMultiplicity(nRawNodes, 0);

      for (auto const& candidate : productionVertexMergeCandidates) {
        if (genVertexCandidateMultiplicity[candidate.first] < UINT16_MAX)
          ++genVertexCandidateMultiplicity[candidate.first];

        if (simVertexCandidateMultiplicity[candidate.second] < UINT16_MAX)
          ++simVertexCandidateMultiplicity[candidate.second];
      }

      for (auto const& candidate : productionVertexMergeCandidates) {
        const uint32_t gv = candidate.first;
        const uint32_t sv = candidate.second;

        // Only accept locally one-to-one GenVertex <-> SimVertex matches.
        // This prevents one busy SimVertex from absorbing many unrelated GenVertices.
        if (genVertexCandidateMultiplicity[gv] != 1 || simVertexCandidateMultiplicity[sv] != 1)
          continue;

        // Do not merge secondary SIM vertices produced by an existing SimTrack.
        // Primary/injection SIM vertices can legitimately have multiple outgoing primary tracks.
        if (rawSimVertexIncomingSimTracks[sv] != 0)
          continue;

        if (rawSimVertexOutgoingSimTracks[sv] == 0)
          continue;

        vertexDSU.unite(rawToVertexTmp[gv], rawToVertexTmp[sv]);
      }
    }

    // ------------------------------------------------------------------
    // 3. Compress particle and vertex representatives
    // ------------------------------------------------------------------
    std::unordered_map<int, uint32_t> particleRepToLogical;
    std::vector<int32_t> rawToParticle(nRawNodes, -1);

    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      if (!keepRawNode[nodeId])
        continue;

      if (rawToParticleTmp[nodeId] >= 0) {
        const int rep = particleDSU.find(rawToParticleTmp[nodeId]);
        auto result = particleRepToLogical.emplace(rep, static_cast<uint32_t>(particleRepToLogical.size()));
        rawToParticle[nodeId] = static_cast<int32_t>(result.first->second);
      }
    }

    std::unordered_map<int, uint32_t> vertexRepToLogical;
    std::vector<int32_t> rawToVertex(nRawNodes, -1);

    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      if (!keepRawNode[nodeId])
        continue;

      if (rawToVertexTmp[nodeId] >= 0) {
        const int rep = vertexDSU.find(rawToVertexTmp[nodeId]);
        auto result = vertexRepToLogical.emplace(rep, static_cast<uint32_t>(vertexRepToLogical.size()));
        rawToVertex[nodeId] = static_cast<int32_t>(result.first->second);
      }
    }

    out->particles.resize(particleRepToLogical.size());
    out->vertices.resize(vertexRepToLogical.size());

    // ------------------------------------------------------------------
    // 4. Fill payload
    // ------------------------------------------------------------------
    for (uint32_t nodeId = 0; nodeId < nRawNodes; ++nodeId) {
      if (!keepRawNode[nodeId])
        continue;

      auto const& ref = raw.nodeRef(nodeId);

      if (rawToParticle[nodeId] >= 0) {
        auto& p = out->particles[static_cast<uint32_t>(rawToParticle[nodeId])];

        if (ref.kind == TruthGraph::NodeKind::GenParticle) {
          p.genNode = static_cast<int32_t>(nodeId);

          if (nodeId < raw.genEventOfNode.size())
            p.genEvent = raw.genEventOfNode[nodeId];

          if (p.pdgId == 0)
            p.pdgId = raw.nodePdgId(nodeId);
          if (p.status == 0)
            p.status = raw.nodeStatus(nodeId);
          if (p.statusFlags == 0)
            p.statusFlags = raw.nodeStatusFlags(nodeId);

          if (haveGenPayload) {
            const int barcode = static_cast<int>(ref.key);
            auto it = genParticlePayload.find(barcode);
            if (it != genParticlePayload.end()) {
              if (p.pdgId == 0)
                p.pdgId = it->second.pdgId;
              if (p.status == 0)
                p.status = it->second.status;

              // Keep the GEN four-momentum as nominal for GEN and GEN+SIM logical particles.
              p.momentum = it->second.momentum;
            }
          }

        } else if (ref.kind == TruthGraph::NodeKind::SimTrack) {
          p.simNode = static_cast<int32_t>(nodeId);

          if (p.pdgId == 0)
            p.pdgId = raw.nodePdgId(nodeId);
          if (p.status == 0)
            p.status = raw.nodeStatus(nodeId);
          if (p.eventId == 0)
            p.eventId = raw.nodeEventId(nodeId);

          if (validHandle(hSimTracks)) {
            const auto trackId = static_cast<uint32_t>(ref.key);
            auto it = simTrackIdToIndex.find(trackId);

            if (it != simTrackIdToIndex.end()) {
              auto const& t = (*hSimTracks)[it->second];

              const math::XYZTLorentzVectorD simMomentum(
                  t.momentum().px(), t.momentum().py(), t.momentum().pz(), t.momentum().e());

              // For SIM-only logical particles, use the SimTrack momentum.
              // For GEN+SIM logical particles, the GEN momentum remains the nominal one.
              if (!p.hasGen()) {
                p.momentum = simMomentum;
              }

              if (t.crossedBoundary()) {
                truth::Checkpoint cp;
                cp.checkpointId = 0;

                const auto& xb = t.getPositionAtBoundary();
                cp.position = math::XYZTLorentzVectorF(xb.x(), xb.y(), xb.z(), xb.t());

                const auto& pb = t.getMomentumAtBoundary();
                cp.momentum = math::XYZTLorentzVectorF(pb.px(), pb.py(), pb.pz(), pb.e());

                p.checkpoints.push_back(cp);
              }
            }
          }
        }
      }

      if (rawToVertex[nodeId] >= 0) {
        auto& v = out->vertices[static_cast<uint32_t>(rawToVertex[nodeId])];

        if (ref.kind == TruthGraph::NodeKind::GenVertex) {
          v.genNode = static_cast<int32_t>(nodeId);

          if (nodeId < raw.genEventOfNode.size())
            v.genEvent = raw.genEventOfNode[nodeId];

          if (haveGenPayload) {
            const int barcode = static_cast<int>(ref.key);
            auto it = genVertexPayload.find(barcode);
            if (it != genVertexPayload.end()) {
              // Keep the GEN position as nominal for GEN and GEN+SIM logical vertices.
              v.position = it->second.position;
            }
          }

        } else if (ref.kind == TruthGraph::NodeKind::SimVertex) {
          v.simNode = static_cast<int32_t>(nodeId);

          if (v.eventId == 0)
            v.eventId = raw.nodeEventId(nodeId);

          if (validHandle(hSimVertices)) {
            const auto simIndex = static_cast<uint32_t>(ref.key);
            if (simIndex < hSimVertices->size()) {
              auto const& sv = (*hSimVertices)[simIndex];
              const auto& pos = sv.position();

              // For SIM-only logical vertices, use the SimVertex position.
              // For GEN+SIM logical vertices, the GEN position remains the nominal one.
              if (!v.hasGen()) {
                v.position = math::XYZTLorentzVectorD(pos.x(), pos.y(), pos.z(), pos.t());
              }
            }
          }
        }
      }
    }

    // ------------------------------------------------------------------
    // 5. Rebuild logical graph
    // ------------------------------------------------------------------
    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t src = 0; src < nRawNodes; ++src) {
      if (!keepRawNode[src])
        continue;

      auto const& srcRef = raw.nodeRef(src);
      const auto dsts = raw.children(src);

      for (uint32_t dst : dsts) {
        if (dst >= nRawNodes || !keepRawNode[dst])
          continue;

        auto const& dstRef = raw.nodeRef(dst);

        if (isVertexKind(srcRef.kind) && isParticleKind(dstRef.kind)) {
          const int32_t lv = rawToVertex[src];
          const int32_t lp = rawToParticle[dst];

          if (lv >= 0 && lp >= 0) {
            vertexToOutgoingParticlePairs.emplace_back(static_cast<uint32_t>(lv), static_cast<uint32_t>(lp));
            particleToProductionVertexPairs.emplace_back(static_cast<uint32_t>(lp), static_cast<uint32_t>(lv));
          }

        } else if (isParticleKind(srcRef.kind) && isVertexKind(dstRef.kind)) {
          const int32_t lp = rawToParticle[src];
          const int32_t lv = rawToVertex[dst];

          if (lp >= 0 && lv >= 0) {
            particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(lp), static_cast<uint32_t>(lv));
            vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(lv), static_cast<uint32_t>(lp));
          }
        }
      }
    }

    buildCSR(
        out->nParticles(), particleToDecayVertexPairs, out->particleToDecayVertexOffsets, out->particleToDecayVertices);

    buildCSR(out->nParticles(),
             particleToProductionVertexPairs,
             out->particleToProductionVertexOffsets,
             out->particleToProductionVertices);

    buildCSR(out->nVertices(),
             vertexToOutgoingParticlePairs,
             out->vertexToOutgoingParticleOffsets,
             out->vertexToOutgoingParticles);

    buildCSR(out->nVertices(),
             vertexToIncomingParticlePairs,
             out->vertexToIncomingParticleOffsets,
             out->vertexToIncomingParticles);

    *out = postProcessGraph(std::move(*out), postProcessing_);

    if (!out->isConsistent()) {
      throw cms::Exception("TruthLogicalGraphProducer") << "Produced truth::Graph is not consistent";
    }

    evt.put(std::move(out));
  }

private:
  edm::EDGetTokenT<TruthGraph> srcToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;
  edm::EDGetTokenT<edm::HepMC3Product> hepmc3Token_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmc2Token_;

  int32_t motherPdgId_;
  bool mergeGenSimVertices_;
  GraphPostProcessingConfig postProcessing_;
};

DEFINE_FWK_MODULE(TruthLogicalGraphProducer);
