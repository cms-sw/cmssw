// Author: Felice Pantaleo - CERN
// Date: 03/2026
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// Legacy HepMC (HepMC2)
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"

// HepMC3
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {

  // Pack EncodedEventId into 64-bit without relying on a particular API
  uint64_t packEventId(EncodedEventId const& id) {
    uint64_t out = 0;
    static_assert(sizeof(EncodedEventId) <= sizeof(uint64_t), "EncodedEventId larger than 64 bits, adjust packing");
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }

  // Disjoint-set union
  struct DSU {
    std::vector<int> p, r;
    explicit DSU(int n) : p(n), r(n, 0) {
      for (int i = 0; i < n; ++i)
        p[i] = i;
    }
    int find(int x) {
      while (p[x] != x) {
        p[x] = p[p[x]];
        x = p[x];
      }
      return x;
    }
    void unite(int a, int b) {
      a = find(a);
      b = find(b);
      if (a == b)
        return;
      if (r[a] < r[b])
        std::swap(a, b);
      p[b] = a;
      if (r[a] == r[b])
        ++r[a];
    }
  };

  // Unique key to index GEN nodes in maps (vertex vs particle + barcode)
  inline int64_t genKeyVertex(int barcode) { return (int64_t(barcode) << 1) | 1LL; }
  inline int64_t genKeyParticle(int barcode) { return (int64_t(barcode) << 1); }

  struct GenBuild {
    // lists of unique barcodes (order of appearance)
    std::vector<int> vtxBarcodes;
    std::vector<int> partBarcodes;

    // index -> barcode for particles in the iteration order (for SimTrack::genpartIndex mapping)
    std::vector<int> particleBarcodeByIndex;

    // bipartite edges in barcode space
    // vtx -> part (outgoing)
    std::vector<std::pair<int, int>> vtxToPart;
    // part -> vtx (end vertex)
    std::vector<std::pair<int, int>> partToVtx;
  };

  // Build GEN info from HepMC2
  GenBuild buildFromHepMC2(HepMC::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    // vertices
    for (auto v = ev.vertices_begin(); v != ev.vertices_end(); ++v) {
      const int vbc = (*v)->barcode();
      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      // outgoing particles
      for (auto po = (*v)->particles_out_const_begin(); po != (*v)->particles_out_const_end(); ++po) {
        const int pbc = (*po)->barcode();
        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);
        gb.vtxToPart.emplace_back(vbc, pbc);
      }

      // incoming particles (orphans are possible; we still model the edge)
      for (auto pi = (*v)->particles_in_const_begin(); pi != (*v)->particles_in_const_end(); ++pi) {
        const int pbc = (*pi)->barcode();
        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);
        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    // particle iteration order -> barcode (for genpartIndex)
    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      gb.particleBarcodeByIndex.push_back((*p)->barcode());
    }

    return gb;
  }

  // Build GEN info from HepMC3
  GenBuild buildFromHepMC3(HepMC3::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    // vertices
    for (auto const& vptr : ev.vertices()) {
      if (!vptr)
        continue;
      const int vbc = vptr->id();  // HepMC3 vertex id acts like barcode; in CMSSW it is typically negative
      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      // outgoing
      for (auto const& po : vptr->particles_out()) {
        if (!po)
          continue;
        const int pbc = po->id();  // particle id (barcode)
        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);
        gb.vtxToPart.emplace_back(vbc, pbc);
      }
      // incoming
      for (auto const& pi : vptr->particles_in()) {
        if (!pi)
          continue;
        const int pbc = pi->id();
        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);
        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    // particle iteration order -> barcode (for genpartIndex)
    gb.particleBarcodeByIndex.reserve(ev.particles().size());
    for (auto const& pptr : ev.particles()) {
      if (!pptr)
        continue;
      gb.particleBarcodeByIndex.push_back(pptr->id());
    }

    return gb;
  }

  template <typename HandleT>
  bool validHandle(HandleT const& h) {
    return h.isValid();
  }

}  // namespace

class TruthGraphProducer : public edm::stream::EDProducer<> {
public:
  explicit TruthGraphProducer(const edm::ParameterSet& cfg)
      : hepmc3Token_(mayConsume<edm::HepMC3Product>(cfg.getParameter<edm::InputTag>("genEventHepMC3"))),
        hepmc2Token_(mayConsume<edm::HepMCProduct>(cfg.getParameter<edm::InputTag>("genEventHepMC"))),
        simTrackToken_(consumes<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
        simVertexToken_(consumes<edm::SimVertexContainer>(cfg.getParameter<edm::InputTag>("simVertices"))),
        addGenToSimEdges_(cfg.getParameter<bool>("addGenToSimEdges")) {
    produces<TruthGraph>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("genEventHepMC3", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMC3Product label (preferred when available)");
    desc.add<edm::InputTag>("genEventHepMC", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMCProduct label (legacy fallback)");

    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"))
        ->setComment("SimTrackContainer label (typically g4SimHits)");
    desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"))
        ->setComment("SimVertexContainer label (typically g4SimHits)");

    desc.add<bool>("addGenToSimEdges", true)
        ->setComment("If true, add cross edges GenParticle -> SimTrack using SimTrack::genpartIndex()");

    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::Event& evt, const edm::EventSetup&) override {
    auto out = std::make_unique<TruthGraph>();

    // --- Fetch SIM
    const auto& simTracks = evt.get(simTrackToken_);
    const auto& simVertices = evt.get(simVertexToken_);

    // --- Fetch GEN (HepMC3 preferred, fallback to HepMC2)
    GenBuild gb;
    bool haveGen = false;

    {
      edm::Handle<edm::HepMC3Product> h3;
      evt.getByToken(hepmc3Token_, h3);
      if (validHandle(h3) && h3->GetEvent() != nullptr) {
        const HepMC3::GenEventData* data = h3->GetEvent();
        HepMC3::GenEvent ev3;
        ev3.read_data(*data);
        gb = buildFromHepMC3(ev3);
        haveGen = true;
      }
    }

    if (!haveGen) {
      edm::Handle<edm::HepMCProduct> h2;
      evt.getByToken(hepmc2Token_, h2);
      if (validHandle(h2) && h2->GetEvent() != nullptr) {
        const HepMC::GenEvent* ev2 = h2->GetEvent();
        gb = buildFromHepMC2(*ev2);
        haveGen = true;
      }
    }

    const uint32_t nSimVtx = static_cast<uint32_t>(simVertices.size());
    const uint32_t nSimTrk = static_cast<uint32_t>(simTracks.size());

    // ----------------------------
    // GEN components -> "GenEvent"
    // ----------------------------
    int nGenEvents = 0;

    std::unordered_map<int64_t, int> tempIndex;  // genKey -> temp idx
    std::vector<int64_t> tempKeys;               // idx -> genKey

    auto getTemp = [&](int64_t k) -> int {
      auto it = tempIndex.find(k);
      if (it != tempIndex.end())
        return it->second;
      const int idx = static_cast<int>(tempKeys.size());
      tempIndex.emplace(k, idx);
      tempKeys.push_back(k);
      return idx;
    };

    std::vector<int> compOfTemp;
    std::unordered_map<int, int> repToComp;

    if (haveGen) {
      for (int vbc : gb.vtxBarcodes)
        (void)getTemp(genKeyVertex(vbc));
      for (int pbc : gb.partBarcodes)
        (void)getTemp(genKeyParticle(pbc));

      DSU dsu(static_cast<int>(tempKeys.size()));

      for (auto const& e : gb.vtxToPart) {
        dsu.unite(getTemp(genKeyVertex(e.first)), getTemp(genKeyParticle(e.second)));
      }
      for (auto const& e : gb.partToVtx) {
        dsu.unite(getTemp(genKeyParticle(e.first)), getTemp(genKeyVertex(e.second)));
      }

      compOfTemp.resize(tempKeys.size(), -1);
      for (int i = 0; i < static_cast<int>(tempKeys.size()); ++i) {
        const int rep = dsu.find(i);
        auto it = repToComp.find(rep);
        if (it == repToComp.end()) {
          const int cid = nGenEvents++;
          repToComp.emplace(rep, cid);
          compOfTemp[i] = cid;
        } else {
          compOfTemp[i] = it->second;
        }
      }

      if (nGenEvents == 0)
        nGenEvents = 1;
    }

    const uint32_t nGenVtx = haveGen ? static_cast<uint32_t>(gb.vtxBarcodes.size()) : 0u;
    const uint32_t nGenPar = haveGen ? static_cast<uint32_t>(gb.partBarcodes.size()) : 0u;

    const uint32_t baseGenEvent = 0;
    const uint32_t baseGenVtx = baseGenEvent + static_cast<uint32_t>(nGenEvents);
    const uint32_t baseGenPar = baseGenVtx + nGenVtx;
    const uint32_t baseSimVtx = baseGenPar + nGenPar;
    const uint32_t baseSimTrk = baseSimVtx + nSimVtx;

    const uint32_t nNodes = baseSimTrk + nSimTrk;

    out->nodes.resize(nNodes);
    out->pdgId.assign(nNodes, 0);
    out->status.assign(nNodes, 0);
    out->eventId.assign(nNodes, 0);

    out->genEventOfNode.assign(nNodes, -1);

    out->simTrackToGen.assign(nNodes, -1);
    out->simTrackToVtx.assign(nNodes, -1);

    // ----------------------------
    // Create GenEvent nodes (roots)
    // ----------------------------
    for (int cid = 0; cid < nGenEvents; ++cid) {
      const uint32_t nodeId = baseGenEvent + static_cast<uint32_t>(cid);
      out->nodes[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenEvent, static_cast<int64_t>(cid)};
      out->eventId[nodeId] = 0;
      out->genEventOfNode[nodeId] = cid;
    }

    // ----------------------------
    // Create GenVertex / GenParticle nodes
    // ----------------------------
    std::unordered_map<int, uint32_t> genVtxBarcodeToNode;
    std::unordered_map<int, uint32_t> genParBarcodeToNode;

    genVtxBarcodeToNode.reserve(nGenVtx * 2);
    genParBarcodeToNode.reserve(nGenPar * 2);

    if (haveGen) {
      for (uint32_t i = 0; i < nGenVtx; ++i) {
        const int vbc = gb.vtxBarcodes[i];
        const uint32_t nodeId = baseGenVtx + i;
        genVtxBarcodeToNode.emplace(vbc, nodeId);
        out->nodes[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenVertex, static_cast<int64_t>(vbc)};
        out->eventId[nodeId] = 0;

        const int tidx = tempIndex.at(genKeyVertex(vbc));
        out->genEventOfNode[nodeId] = compOfTemp[tidx];
      }

      for (uint32_t i = 0; i < nGenPar; ++i) {
        const int pbc = gb.partBarcodes[i];
        const uint32_t nodeId = baseGenPar + i;
        genParBarcodeToNode.emplace(pbc, nodeId);
        out->nodes[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenParticle, static_cast<int64_t>(pbc)};
        out->eventId[nodeId] = 0;

        const int tidx = tempIndex.at(genKeyParticle(pbc));
        out->genEventOfNode[nodeId] = compOfTemp[tidx];
      }
    }

    // GenParticle barcode -> production GenVertex barcode (from vtx -> part)
    std::unordered_map<int, int> genParToProdVtx;
    if (haveGen) {
      genParToProdVtx.reserve(gb.partBarcodes.size() * 2);
      for (auto const& vp : gb.vtxToPart) {
        genParToProdVtx.emplace(vp.second, vp.first);
      }
    }

    // ----------------------------
    // Create SimVertex nodes
    // ----------------------------
    std::vector<uint32_t> simVtxIndexToNode(nSimVtx, 0);
    for (uint32_t i = 0; i < nSimVtx; ++i) {
      const uint32_t nodeId = baseSimVtx + i;
      simVtxIndexToNode[i] = nodeId;
      out->nodes[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(i)};
      out->eventId[nodeId] = packEventId(simVertices[i].eventId());
    }

    // ----------------------------
    // Create SimTrack nodes
    // ----------------------------
    std::unordered_map<uint32_t, uint32_t> simTrackIdToNode;
    simTrackIdToNode.reserve(nSimTrk * 2);

    for (uint32_t i = 0; i < nSimTrk; ++i) {
      const uint32_t nodeId = baseSimTrk + i;
      const uint32_t tid = simTracks[i].trackId();
      simTrackIdToNode.emplace(tid, nodeId);

      out->nodes[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(tid)};
      out->pdgId[nodeId] = simTracks[i].type();
      out->eventId[nodeId] = packEventId(simTracks[i].eventId());

      const int vtxIdx = simTracks[i].vertIndex();
      if (vtxIdx >= 0 && static_cast<uint32_t>(vtxIdx) < nSimVtx) {
        out->simTrackToVtx[nodeId] = static_cast<int32_t>(simVtxIndexToNode[static_cast<uint32_t>(vtxIdx)]);
      }

      if (haveGen && !simTracks[i].noGenpart()) {
        const int ig = simTracks[i].genpartIndex();
        if (ig >= 0 && static_cast<uint32_t>(ig) < gb.particleBarcodeByIndex.size()) {
          const int barcode = gb.particleBarcodeByIndex[static_cast<uint32_t>(ig)];
          auto it = genParBarcodeToNode.find(barcode);
          if (it != genParBarcodeToNode.end()) {
            out->simTrackToGen[nodeId] = static_cast<int32_t>(it->second);
          }
        }
      }
    }

    // ----------------------------
    // Build edges (+ kinds) and compress to CSR
    // ----------------------------
    std::vector<std::pair<uint32_t, uint32_t>> edgePairs;
    std::vector<uint8_t> edgeKinds;
    edgePairs.reserve(12 * (nGenVtx + nGenPar + nSimTrk));
    edgeKinds.reserve(edgePairs.capacity());

    auto push_edge = [&](uint32_t src, uint32_t dst, TruthGraph::EdgeKind k) {
      edgePairs.emplace_back(src, dst);
      edgeKinds.emplace_back(static_cast<uint8_t>(k));
    };

    // Dedup GenVtx->SimVtx cross edges
    std::unordered_set<uint64_t> genVtxToSimVtxSeen;
    genVtxToSimVtxSeen.reserve(2 * nSimTrk);
    auto packPair = [](uint32_t a, uint32_t b) -> uint64_t { return (uint64_t(a) << 32) | uint64_t(b); };

    // GEN edges: connect GenEvent(component) -> GenVertex roots (or all vertices if no roots found)
    if (haveGen) {
      std::unordered_map<int, int> vtxIncoming;
      vtxIncoming.reserve(nGenVtx * 2);
      for (int vbc : gb.vtxBarcodes)
        vtxIncoming.emplace(vbc, 0);
      for (auto const& e : gb.partToVtx) {
        auto it = vtxIncoming.find(e.second);
        if (it != vtxIncoming.end())
          it->second++;
      }

      std::vector<std::vector<int>> rootsByComp(nGenEvents);
      std::vector<std::vector<int>> allVtxByComp(nGenEvents);

      for (int vbc : gb.vtxBarcodes) {
        const int tidx = tempIndex.at(genKeyVertex(vbc));
        const int cid = compOfTemp[tidx];
        if (cid < 0 || cid >= nGenEvents)
          continue;
        allVtxByComp[cid].push_back(vbc);
        if (vtxIncoming[vbc] == 0)
          rootsByComp[cid].push_back(vbc);
      }

      for (int cid = 0; cid < nGenEvents; ++cid) {
        const uint32_t genEventNode = baseGenEvent + static_cast<uint32_t>(cid);
        auto roots = rootsByComp[cid];
        if (roots.empty())
          roots = allVtxByComp[cid];
        for (int vbc : roots) {
          auto itV = genVtxBarcodeToNode.find(vbc);
          if (itV != genVtxBarcodeToNode.end()) {
            push_edge(genEventNode, itV->second, TruthGraph::EdgeKind::Gen);
          }
        }
      }

      for (auto const& e : gb.vtxToPart) {
        auto itV = genVtxBarcodeToNode.find(e.first);
        auto itP = genParBarcodeToNode.find(e.second);
        if (itV != genVtxBarcodeToNode.end() && itP != genParBarcodeToNode.end()) {
          push_edge(itV->second, itP->second, TruthGraph::EdgeKind::Gen);
        }
      }
      for (auto const& e : gb.partToVtx) {
        auto itP = genParBarcodeToNode.find(e.first);
        auto itV = genVtxBarcodeToNode.find(e.second);
        if (itP != genParBarcodeToNode.end() && itV != genVtxBarcodeToNode.end()) {
          push_edge(itP->second, itV->second, TruthGraph::EdgeKind::Gen);
        }
      }
    }

    // SIM edges:
    for (uint32_t i = 0; i < nSimTrk; ++i) {
      const auto& t = simTracks[i];
      const uint32_t childNode = baseSimTrk + i;

      const int vtxIdx = t.vertIndex();
      if (vtxIdx < 0 || static_cast<uint32_t>(vtxIdx) >= nSimVtx)
        continue;
      const uint32_t vtxNode = simVtxIndexToNode[static_cast<uint32_t>(vtxIdx)];

      push_edge(vtxNode, childNode, TruthGraph::EdgeKind::Sim);

      const int parentTid = simVertices[static_cast<uint32_t>(vtxIdx)].parentIndex();  // trackId (not index)
      if (parentTid > 0) {
        auto itParent = simTrackIdToNode.find(static_cast<uint32_t>(parentTid));
        if (itParent != simTrackIdToNode.end()) {
          push_edge(itParent->second, vtxNode, TruthGraph::EdgeKind::Sim);
        }
      }
    }

    // CROSS GEN->SIM: GenParticle -> SimTrack
    if (addGenToSimEdges_ && haveGen) {
      for (uint32_t i = 0; i < nSimTrk; ++i) {
        const uint32_t simNode = baseSimTrk + i;
        const int32_t genNode = out->simTrackToGen[simNode];
        if (genNode >= 0) {
          push_edge(static_cast<uint32_t>(genNode), simNode, TruthGraph::EdgeKind::GenToSim);
        }
      }
    }

    // CROSS GEN->SIM: GenVertex(prod of GenParticle) -> SimVertex(prod of SimTrack)
    if (haveGen) {
      for (uint32_t i = 0; i < nSimTrk; ++i) {
        const uint32_t simTrackNode = baseSimTrk + i;

        const int32_t genParNode = out->simTrackToGen[simTrackNode];
        if (genParNode < 0)
          continue;

        const int32_t simVtxNode_i32 = out->simTrackToVtx[simTrackNode];
        if (simVtxNode_i32 < 0)
          continue;
        const uint32_t simVtxNode = static_cast<uint32_t>(simVtxNode_i32);

        auto const& genParRef = out->nodes[static_cast<uint32_t>(genParNode)];
        if (genParRef.kind != TruthGraph::NodeKind::GenParticle)
          continue;
        const int pbc = static_cast<int>(genParRef.key);

        auto itProd = genParToProdVtx.find(pbc);
        if (itProd == genParToProdVtx.end())
          continue;

        auto itV = genVtxBarcodeToNode.find(itProd->second);
        if (itV == genVtxBarcodeToNode.end())
          continue;

        const uint32_t genVtxNode = itV->second;

        const uint64_t packed = packPair(genVtxNode, simVtxNode);
        if (!genVtxToSimVtxSeen.insert(packed).second)
          continue;

        push_edge(genVtxNode, simVtxNode, TruthGraph::EdgeKind::GenToSim);
      }
    }

    // CSR compress
    out->offsets.assign(nNodes + 1, 0);
    for (auto const& e : edgePairs) {
      if (e.first < nNodes)
        out->offsets[e.first + 1] += 1;
    }
    for (uint32_t i = 1; i <= nNodes; ++i)
      out->offsets[i] += out->offsets[i - 1];

    const uint32_t nEdges = out->offsets.back();
    out->edges.assign(nEdges, 0);
    out->edgeKind.assign(nEdges, static_cast<uint8_t>(TruthGraph::EdgeKind::Gen));

    std::vector<uint32_t> cursor = out->offsets;
    for (size_t i = 0; i < edgePairs.size(); ++i) {
      const uint32_t src = edgePairs[i].first;
      const uint32_t dst = edgePairs[i].second;
      if (src < nNodes && dst < nNodes) {
        const uint32_t pos = cursor[src]++;
        out->edges[pos] = dst;
        out->edgeKind[pos] = edgeKinds[i];
      }
    }

    evt.put(std::move(out));
  }

private:
  edm::EDGetTokenT<edm::HepMC3Product> hepmc3Token_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmc2Token_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;
  bool addGenToSimEdges_;
};

DEFINE_FWK_MODULE(TruthGraphProducer);
