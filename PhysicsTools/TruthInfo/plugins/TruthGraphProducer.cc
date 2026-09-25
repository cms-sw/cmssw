// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// Author: Felice Pantaleo - CERN
// Date: 03/2026

#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// Legacy HepMC, HepMC2.
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

// HepMC3.
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

using truth::buildFromHepMC2;
using truth::buildFromHepMC3;
using truth::GenBuild;
using truth::genKeyParticle;
using truth::genKeyVertex;

namespace {

  // Pack EncodedEventId into 64 bit without relying on a particular public API.
  uint64_t packEventId(EncodedEventId const& id) {
    uint64_t out = 0;
    static_assert(sizeof(EncodedEventId) <= sizeof(uint64_t), "EncodedEventId larger than 64 bits, adjust packing");
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }

  struct DSU {
    std::vector<int> p;
    std::vector<int> r;

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
        addGenToSimEdges_(cfg.getParameter<bool>("addGenToSimEdges")),
        collapseGenShower_(cfg.getParameter<bool>("collapseGenShower")) {
    produces<TruthGraph>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("genEventHepMC3", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMC3Product label, preferred when available");
    desc.add<edm::InputTag>("genEventHepMC", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMCProduct label, legacy fallback");

    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"))
        ->setComment("SimTrackContainer label, typically g4SimHits");
    desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"))
        ->setComment("SimVertexContainer label, typically g4SimHits");

    desc.add<bool>("addGenToSimEdges", true)
        ->setComment(
            "If true, add GenParticle -> SimTrack cross edges. The association is built only for primary "
            "SimTracks, interpreting SimTrack::genpartIndex() as a HepMC barcode.");

    desc.add<bool>("collapseGenShower", true)
        ->setComment(
            "If true, contract the GEN parton shower and the intermediate copies of a resonance, keeping ancestry. "
            "A GEN particle survives if a SimTrack continues it, or it is stable, or it is flagged isHardProcess, "
            "or it is the last copy of something that is not a parton, diquark, string, cluster or beam "
            "pseudoparticle.");

    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::Event& evt, const edm::EventSetup&) override {
    auto out = std::make_unique<TruthGraph>();

    const auto& simTracks = evt.get(simTrackToken_);
    const auto& simVertices = evt.get(simVertexToken_);

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
        gb = buildFromHepMC2(*h2->GetEvent());
        haveGen = true;
      }
    }

    if (haveGen && collapseGenShower_)
      if (!truth::collapseGenShower(gb, truth::simContinuedGenBarcodes(simTracks)) && !degradedCollapseWarned_) {
        // Sample-level condition; once per stream is the whole message.
        degradedCollapseWarned_ = true;
        edm::LogWarning("TruthGraphProducer")
            << "collapseGenShower ran on a GEN record with no packed status flags, which "
               "buildFromHepMC3 does not fill. The isHardProcess and isLastCopy keep rules "
               "are then dead and every intermediate resonance is dropped, so a selection "
               "preset seeded on a resonance pdgId will match nothing. Set "
               "collapseGenShower=False on a HepMC3 sample.";
      }

    const uint32_t nSimVtx = static_cast<uint32_t>(simVertices.size());
    const uint32_t nSimTrk = static_cast<uint32_t>(simTracks.size());

    int nGenEvents = 0;

    std::unordered_map<int64_t, int> tempIndex;
    std::vector<int64_t> tempKeys;

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

    out->nodes().resize(nNodes);

    out->pdgId().assign(nNodes, 0);
    out->status().assign(nNodes, 0);
    out->eventId().assign(nNodes, 0);
    out->statusFlags().assign(nNodes, 0);
    out->genEventOfNode().assign(nNodes, -1);
    out->simVertexProcessType().assign(nNodes, 0);
    out->simTrackBackscattered().assign(nNodes, 0);

    out->simTrackToGen().assign(nNodes, -1);
    out->simTrackToVtx().assign(nNodes, -1);
    out->simVtxToGen().assign(nNodes, -1);

    for (int cid = 0; cid < nGenEvents; ++cid) {
      const uint32_t nodeId = baseGenEvent + static_cast<uint32_t>(cid);

      out->nodes()[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenEvent, static_cast<int64_t>(cid)};
      out->eventId()[nodeId] = 0;
      out->genEventOfNode()[nodeId] = cid;
    }

    std::unordered_map<int, uint32_t> genVtxBarcodeToNode;
    std::unordered_map<int, uint32_t> genParBarcodeToNode;

    genVtxBarcodeToNode.reserve(nGenVtx * 2);
    genParBarcodeToNode.reserve(nGenPar * 2);

    if (haveGen) {
      for (uint32_t i = 0; i < nGenVtx; ++i) {
        const int vbc = gb.vtxBarcodes[i];
        const uint32_t nodeId = baseGenVtx + i;

        genVtxBarcodeToNode.emplace(vbc, nodeId);

        out->nodes()[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenVertex, static_cast<int64_t>(vbc)};
        out->eventId()[nodeId] = 0;

        const int tidx = tempIndex.at(genKeyVertex(vbc));
        out->genEventOfNode()[nodeId] = compOfTemp[tidx];
      }

      for (uint32_t i = 0; i < nGenPar; ++i) {
        const int pbc = gb.partBarcodes[i];
        const uint32_t nodeId = baseGenPar + i;

        genParBarcodeToNode.emplace(pbc, nodeId);

        out->nodes()[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::GenParticle, static_cast<int64_t>(pbc)};
        out->eventId()[nodeId] = 0;

        const int tidx = tempIndex.at(genKeyParticle(pbc));
        out->genEventOfNode()[nodeId] = compOfTemp[tidx];

        auto itPdg = gb.particlePdgIdByBarcode.find(pbc);
        if (itPdg != gb.particlePdgIdByBarcode.end())
          out->pdgId()[nodeId] = itPdg->second;

        auto itStatus = gb.particleStatusByBarcode.find(pbc);
        if (itStatus != gb.particleStatusByBarcode.end())
          out->status()[nodeId] = itStatus->second;

        // Computed from the HepMC record via MCTruthHelper at build time (HepMC2);
        // 0 when unavailable (HepMC3 path, missing barcode).
        auto itFlags = gb.particleStatusFlagsByBarcode.find(pbc);
        out->statusFlags()[nodeId] = (itFlags != gb.particleStatusFlagsByBarcode.end()) ? itFlags->second : 0;
      }
    }

    // Map each GEN particle barcode to its production GenVertex barcode.
    // gb.vtxToPart holds (vertex barcode -> outgoing particle barcode), i.e. the
    // production vertex of each outgoing particle.
    std::unordered_map<int, int> genPartToProdVtxBarcode;
    if (haveGen) {
      genPartToProdVtxBarcode.reserve(gb.vtxToPart.size() * 2);
      for (auto const& e : gb.vtxToPart)
        genPartToProdVtxBarcode.emplace(e.second, e.first);
    }

    std::vector<uint32_t> simVtxIndexToNode(nSimVtx, 0);

    for (uint32_t i = 0; i < nSimVtx; ++i) {
      const uint32_t nodeId = baseSimVtx + i;

      simVtxIndexToNode[i] = nodeId;

      out->nodes()[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimVertex, static_cast<int64_t>(i)};
      out->eventId()[nodeId] = packEventId(simVertices[i].eventId());
      out->simVertexProcessType()[nodeId] = static_cast<uint16_t>(simVertices[i].processType());
    }

    std::unordered_map<uint32_t, uint32_t> simTrackIdToNode;
    simTrackIdToNode.reserve(nSimTrk * 2);

    for (uint32_t i = 0; i < nSimTrk; ++i) {
      auto const& simTrack = simTracks[i];

      const uint32_t nodeId = baseSimTrk + i;
      const uint32_t tid = simTrack.trackId();

      simTrackIdToNode.emplace(tid, nodeId);

      out->nodes()[nodeId] = TruthGraph::NodeRef{TruthGraph::NodeKind::SimTrack, static_cast<int64_t>(tid)};
      out->pdgId()[nodeId] = simTrack.type();
      out->eventId()[nodeId] = packEventId(simTrack.eventId());
      out->simTrackBackscattered()[nodeId] = simTrack.isFromBackScattering() ? 1 : 0;

      const int vtxIdx = simTrack.vertIndex();
      if (vtxIdx >= 0 && static_cast<uint32_t>(vtxIdx) < nSimVtx) {
        out->simTrackToVtx()[nodeId] = static_cast<int32_t>(simVtxIndexToNode[static_cast<uint32_t>(vtxIdx)]);
      }

      // SimTrack::genpartIndex() must be used only for primary G4 tracks.
      // For non-primary tracks, getPrimaryOrLastStoredID() can still contain
      // a generator barcode, but that is ancestry information for orphan or
      // backscattered tracks, not a direct SimTrack -> GenParticle association.
      if (addGenToSimEdges_ && haveGen && simTrack.isPrimary()) {
        const int barcode = simTrack.genpartIndex();

        if (barcode != -1) {
          auto it = genParBarcodeToNode.find(barcode);

          if (it != genParBarcodeToNode.end()) {
            const int simPdgId = simTrack.type();
            const int genPdgId = out->nodePdgId(it->second);

            if (genPdgId == 0 || genPdgId == simPdgId) {
              out->simTrackToGen()[nodeId] = static_cast<int32_t>(it->second);

              // Provenance SimVertex -> GenVertex association: the SimTrack's production
              // SimVertex corresponds to the production GenVertex of its GenParticle.
              const int32_t simVtxNode = out->simTrackToVtx()[nodeId];
              if (simVtxNode >= 0) {
                auto itProd = genPartToProdVtxBarcode.find(barcode);
                if (itProd != genPartToProdVtxBarcode.end()) {
                  auto itGV = genVtxBarcodeToNode.find(itProd->second);
                  if (itGV != genVtxBarcodeToNode.end()) {
                    const int32_t gvNode = static_cast<int32_t>(itGV->second);
                    int32_t& slot = out->simVtxToGen()[simVtxNode];
                    if (slot < 0) {
                      slot = gvNode;
                    } else if (slot != gvNode) {
                      edm::LogPrint("TruthGraphProducer")
                          << "SimVertex node " << simVtxNode << " associated to multiple GenVertex nodes (" << slot
                          << " and " << gvNode << "); keeping the first";
                    }
                  }
                }
              }
            } else {
              edm::LogPrint("TruthGraphProducer")
                  << "Rejecting primary SimTrack->GenParticle association with mismatched PDG id: "
                  << "simTrack index=" << i << " trackId=" << simTrack.trackId() << " genBarcode=" << barcode
                  << " simPdgId=" << simPdgId << " genNode=" << it->second << " genPdgId=" << genPdgId;
            }
          } else {
            edm::LogPrint("TruthGraphProducer")
                << "Rejecting primary SimTrack->GenParticle association with missing GEN barcode: "
                << "simTrack index=" << i << " trackId=" << simTrack.trackId() << " genBarcode=" << barcode;
          }
        }
      }
    }

    std::vector<std::pair<uint32_t, uint32_t>> edgePairs;
    std::vector<uint8_t> edgeKinds;

    edgePairs.reserve(8 * (nGenVtx + nGenPar + nSimTrk));
    edgeKinds.reserve(edgePairs.capacity());

    auto push_edge = [&](uint32_t src, uint32_t dst, TruthGraph::EdgeKind k) {
      edgePairs.emplace_back(src, dst);
      edgeKinds.emplace_back(static_cast<uint8_t>(k));
    };

    if (haveGen) {
      std::unordered_map<int, int> vtxIncoming;
      vtxIncoming.reserve(nGenVtx * 2);

      for (int vbc : gb.vtxBarcodes)
        vtxIncoming.emplace(vbc, 0);

      for (auto const& e : gb.partToVtx) {
        auto it = vtxIncoming.find(e.second);
        if (it != vtxIncoming.end())
          ++it->second;
      }

      // Residual gap, shared with TruthGraphAccumulator so the two stay consistent: source
      // counting is per undirected component, but reachability from the GenEvent node is
      // DIRECTED. A component containing both a true source and a beam-fed branch would
      // attach only the source and leave the branch unreachable. No current record mixes
      // the two in one component: a collider record is wholly sourceless and a gun record
      // wholly source-rooted.
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

    // Production edges: SimVertex -> outgoing SimTrack, one per track.
    for (uint32_t i = 0; i < nSimTrk; ++i) {
      auto const& simTrack = simTracks[i];

      const uint32_t childNode = baseSimTrk + i;

      const int vtxIdx = simTrack.vertIndex();
      if (vtxIdx < 0 || static_cast<uint32_t>(vtxIdx) >= nSimVtx)
        continue;

      const uint32_t vtxNode = simVtxIndexToNode[static_cast<uint32_t>(vtxIdx)];

      push_edge(vtxNode, childNode, TruthGraph::EdgeKind::Sim);
    }

    // Decay edges: parent SimTrack -> SimVertex, one per vertex. Built in a
    // separate pass over SimVertices (not inside the track loop) so the edge is
    // emitted once per vertex rather than once per outgoing daughter, which would
    // duplicate parentTrack -> vertex by the vertex's out-degree.
    for (uint32_t i = 0; i < nSimVtx; ++i) {
      const int parentTid = simVertices[i].parentIndex();
      if (parentTid <= 0)
        continue;

      auto itParent = simTrackIdToNode.find(static_cast<uint32_t>(parentTid));
      if (itParent != simTrackIdToNode.end()) {
        push_edge(itParent->second, simVtxIndexToNode[i], TruthGraph::EdgeKind::Sim);
      }
    }

    // Cross-domain particle associations only. These edges are created only for
    // primary SimTracks that carry a validated HepMC barcode.
    //
    // GenVertex -> SimVertex edges are intentionally not created here because
    // shared Geant4 source or injection vertices can create artificial many-to-one topology.
    if (addGenToSimEdges_ && haveGen) {
      for (uint32_t i = 0; i < nSimTrk; ++i) {
        const uint32_t simNode = baseSimTrk + i;
        const int32_t genNode = out->simTrackToGen()[simNode];

        if (genNode >= 0) {
          push_edge(static_cast<uint32_t>(genNode), simNode, TruthGraph::EdgeKind::GenToSim);
        }
      }

      // SimVertex -> GenVertex provenance edges. Unlike the GenVertex -> SimVertex
      // direction warned about above, these are derived from per-track primary
      // associations and stored as a single edge per SimVertex (simVtxToGen).
      for (uint32_t i = 0; i < nSimVtx; ++i) {
        const uint32_t simVtxNode = baseSimVtx + i;
        const int32_t genVtxNode = out->simVtxToGen()[simVtxNode];

        if (genVtxNode >= 0) {
          push_edge(simVtxNode, static_cast<uint32_t>(genVtxNode), TruthGraph::EdgeKind::SimToGen);
        }
      }
    }

    out->offsets().assign(nNodes + 1, 0);

    for (auto const& e : edgePairs) {
      if (e.first < nNodes)
        ++out->offsets()[e.first + 1];
    }

    for (uint32_t i = 1; i <= nNodes; ++i)
      out->offsets()[i] += out->offsets()[i - 1];

    const uint32_t nEdges = out->offsets().back();

    out->edges().assign(nEdges, 0);
    out->edgeKind().assign(nEdges, static_cast<uint8_t>(TruthGraph::EdgeKind::Gen));

    std::vector<uint32_t> cursor = out->offsets();

    for (std::size_t i = 0; i < edgePairs.size(); ++i) {
      const uint32_t src = edgePairs[i].first;
      const uint32_t dst = edgePairs[i].second;

      if (src < nNodes && dst < nNodes) {
        const uint32_t pos = cursor[src]++;
        out->edges()[pos] = dst;
        out->edgeKind()[pos] = edgeKinds[i];
      }
    }

    unsigned nGenEventOut = 0;
    unsigned nGenVertexOut = 0;
    unsigned nGenParticleOut = 0;
    unsigned nSimVertexOut = 0;
    unsigned nSimTrackOut = 0;
    unsigned nGenToSimParticleLinks = 0;
    unsigned nSimVtxToGenLinks = 0;

    for (uint32_t i = 0; i < out->nNodes(); ++i) {
      switch (out->nodeRef(i).kind) {
        case TruthGraph::NodeKind::GenEvent:
          ++nGenEventOut;
          break;
        case TruthGraph::NodeKind::GenVertex:
          ++nGenVertexOut;
          break;
        case TruthGraph::NodeKind::GenParticle:
          ++nGenParticleOut;
          break;
        case TruthGraph::NodeKind::SimVertex:
          ++nSimVertexOut;
          if (out->simVtxToGen()[i] >= 0)
            ++nSimVtxToGenLinks;
          break;
        case TruthGraph::NodeKind::SimTrack:
          ++nSimTrackOut;
          if (out->simTrackToGen()[i] >= 0)
            ++nGenToSimParticleLinks;
          break;
      }
    }

    edm::LogPrint("TruthGraphProducer") << "TruthGraph nodes: "
                                        << "GenEvent=" << nGenEventOut << " GenVertex=" << nGenVertexOut
                                        << " GenParticle=" << nGenParticleOut << " SimVertex=" << nSimVertexOut
                                        << " SimTrack=" << nSimTrackOut << " total=" << out->nNodes()
                                        << " edges=" << out->nEdges()
                                        << " primaryGenToSimParticleLinks=" << nGenToSimParticleLinks
                                        << " simVtxToGenVertexLinks=" << nSimVtxToGenLinks;

    evt.put(std::move(out));
  }

private:
  edm::EDGetTokenT<edm::HepMC3Product> hepmc3Token_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmc2Token_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;

  bool addGenToSimEdges_;
  bool collapseGenShower_;
  bool degradedCollapseWarned_ = false;
};

DEFINE_FWK_MODULE(TruthGraphProducer);
