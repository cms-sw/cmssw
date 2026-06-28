// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {

  std::string pdgNameUtf8(int pdgId) {
    const int ap = std::abs(pdgId);

    if (pdgId == 11)
      return "e-";
    if (pdgId == -11)
      return "e+";
    if (pdgId == 13)
      return "mu-";
    if (pdgId == -13)
      return "mu+";
    if (pdgId == 15)
      return "tau-";
    if (pdgId == -15)
      return "tau+";

    if (pdgId == 12)
      return "nu_e";
    if (pdgId == -12)
      return "anti-nu_e";
    if (pdgId == 14)
      return "nu_mu";
    if (pdgId == -14)
      return "anti-nu_mu";
    if (pdgId == 16)
      return "nu_tau";
    if (pdgId == -16)
      return "anti-nu_tau";

    if (pdgId == 22)
      return "gamma";
    if (pdgId == 21)
      return "g";
    if (pdgId == 23)
      return "Z0";
    if (pdgId == 24)
      return "W+";
    if (pdgId == -24)
      return "W-";
    if (pdgId == 25)
      return "H";

    if (pdgId == 2212)
      return "p";
    if (pdgId == -2212)
      return "anti-p";
    if (pdgId == 2112)
      return "n";
    if (pdgId == -2112)
      return "anti-n";

    if (pdgId == 111)
      return "pi0";
    if (pdgId == 211)
      return "pi+";
    if (pdgId == -211)
      return "pi-";
    if (pdgId == 321)
      return "K+";
    if (pdgId == -321)
      return "K-";
    if (pdgId == 130)
      return "K0_L";
    if (pdgId == 310)
      return "K0_S";

    if (ap >= 1 && ap <= 6) {
      static const char* qname[7] = {"", "d", "u", "s", "c", "b", "t"};
      std::string s = qname[ap];
      if (pdgId < 0)
        s = "anti-" + s;
      return s;
    }

    return "pdg";
  }

  std::string pdgLabel(int pdgId) {
    std::ostringstream ss;
    const std::string name = pdgNameUtf8(pdgId);
    if (name == "pdg")
      ss << "pdg(" << pdgId << ")";
    else
      ss << name << " (" << pdgId << ")";
    return ss.str();
  }

  const char* rawKindName(TruthGraph::NodeKind k) {
    switch (k) {
      case TruthGraph::NodeKind::GenEvent:
        return "GenEvent";
      case TruthGraph::NodeKind::GenVertex:
        return "GenVertex";
      case TruthGraph::NodeKind::GenParticle:
        return "GenParticle";
      case TruthGraph::NodeKind::SimVertex:
        return "SimVertex";
      case TruthGraph::NodeKind::SimTrack:
        return "SimTrack";
    }
    return "Unknown";
  }

  std::string rawNodeSummary(TruthGraph const* raw, int32_t nodeId) {
    if (raw == nullptr || nodeId < 0 || static_cast<uint32_t>(nodeId) >= raw->nNodes())
      return "n/a";

    auto const& r = raw->nodeRef(static_cast<uint32_t>(nodeId));

    std::ostringstream ss;
    ss << rawKindName(r.kind) << " #" << nodeId << " key=" << r.key;
    return ss.str();
  }

  template <typename X4>
  std::string fmtX4(X4 const& x4) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << "(" << x4.x() << ", " << x4.y() << ", " << x4.z() << ", " << x4.t() << ")";
    return ss.str();
  }

  template <typename P4>
  std::string fmtP4(P4 const& p4) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << "(" << p4.px() << ", " << p4.py() << ", " << p4.pz() << ", " << p4.e() << ")";
    return ss.str();
  }

  const char* logicalVertexDomain(truth::VertexData const& d) {
    // Artificial source vertices (Interaction / Upstream-ISR / UnderlyingEvent)
    // have no GEN or SIM back-reference by construction; they are graph-internal
    // bookkeeping nodes, so they get their own "Internal" domain rather than
    // looking like an unclassified real vertex.
    if (d.isArtificial())
      return "Internal";
    if (d.hasGen() && !d.hasSim())
      return "GEN";
    if (!d.hasGen() && d.hasSim())
      return "SIM";
    if (d.hasGen() && d.hasSim())
      return "GEN+SIM";
    return "UNKNOWN";
  }

  std::string statusFlagsLabel(uint16_t flags) {
    struct FlagInfo {
      uint16_t bit;
      const char* name;
    };

    static constexpr FlagInfo flagInfos[] = {
        {1u << 0, "isPrompt"},
        {1u << 1, "isDecayedLeptonHadron"},
        {1u << 2, "isTauDecayProduct"},
        {1u << 3, "isPromptTauDecayProduct"},
        {1u << 4, "isDirectTauDecayProduct"},
        {1u << 5, "isDirectPromptTauDecayProduct"},
        {1u << 6, "isDirectHadronDecayProduct"},
        {1u << 7, "isHardProcess"},
        {1u << 8, "fromHardProcess"},
        {1u << 9, "isHardProcessTauDecayProduct"},
        {1u << 10, "isDirectHardProcessTauDecayProduct"},
        {1u << 11, "fromHardProcessBeforeFSR"},
        {1u << 12, "isFirstCopy"},
        {1u << 13, "isLastCopy"},
        {1u << 14, "isLastCopyBeforeFSR"},
    };

    std::ostringstream ss;
    bool first = true;

    for (auto const& flag : flagInfos) {
      if ((flags & flag.bit) == 0)
        continue;

      if (!first)
        ss << ", ";
      ss << flag.name;
      first = false;
    }

    if (first)
      return "none";

    return ss.str();
  }

  std::string fmtEnergy(float energy) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(6) << energy;
    return ss.str();
  }

  struct HitSummary {
    uint32_t nSimHits = 0;
    uint32_t nMatchedRecHits = 0;
    uint32_t nMissingRecHits = 0;
    float simHitEnergy = 0.f;
    float recHitEnergy = 0.f;
  };

  HitSummary summarizeHits(std::span<const truth::LogicalGraphHitIndex::Hit> hits,
                           std::vector<float> const& recHitEnergies) {
    HitSummary summary;
    summary.nSimHits = static_cast<uint32_t>(hits.size());

    for (auto const& hit : hits) {
      summary.simHitEnergy += hit.energy;

      if (hit.recHitIndex == truth::LogicalGraphHitIndex::Hit::kInvalidRecHitIndex ||
          hit.recHitIndex >= recHitEnergies.size()) {
        ++summary.nMissingRecHits;
        continue;
      }

      ++summary.nMatchedRecHits;
      summary.recHitEnergy += recHitEnergies[hit.recHitIndex];
    }

    return summary;
  }

  template <typename F>
  void forEachDescendantParticle(truth::Graph const& g, uint32_t particleId, F&& f) {
    if (particleId >= g.nParticles())
      return;

    // Iterative DFS with a visited set: cycle-safe (a stray cycle would otherwise
    // overflow the stack) and visits each descendant exactly once (no exponential
    // blow-up on re-convergent / diamond topologies). f is invoked once per
    // distinct descendant, never on the start particle itself.
    std::vector<uint8_t> visited(g.nParticles(), 0);
    std::vector<uint32_t> stack;
    visited[particleId] = 1;
    stack.push_back(particleId);

    while (!stack.empty()) {
      const uint32_t current = stack.back();
      stack.pop_back();

      for (const uint32_t vertexId : g.decayVertices(current)) {
        if (vertexId >= g.nVertices())
          continue;

        for (const uint32_t childId : g.outgoingParticles(vertexId)) {
          if (childId >= g.nParticles() || visited[childId])
            continue;
          visited[childId] = 1;
          f(childId);
          stack.push_back(childId);
        }
      }
    }
  }

  bool hasVisibleIncomingParticle(truth::Graph const& g, uint32_t vertexId, std::vector<uint8_t> const& hideParticle) {
    for (const uint32_t p : g.incomingParticles(vertexId)) {
      if (p < hideParticle.size() && !hideParticle[p])
        return true;
    }

    return false;
  }

  bool hasVisibleOutgoingParticle(truth::Graph const& g, uint32_t vertexId, std::vector<uint8_t> const& hideParticle) {
    for (const uint32_t p : g.outgoingParticles(vertexId)) {
      if (p < hideParticle.size() && !hideParticle[p])
        return true;
    }

    return false;
  }

  bool shouldHideVertexAfterParticleFiltering(truth::Graph const& g,
                                              uint32_t vertexId,
                                              std::vector<uint8_t> const& hideParticle) {
    const bool hasIncoming = !g.incomingParticles(vertexId).empty();
    const bool hasOutgoing = !g.outgoingParticles(vertexId).empty();

    const bool hasVisibleIncoming = hasVisibleIncomingParticle(g, vertexId, hideParticle);
    const bool hasVisibleOutgoing = hasVisibleOutgoingParticle(g, vertexId, hideParticle);

    // Hide vertices fully disconnected by the particle filter.
    if (!hasVisibleIncoming && !hasVisibleOutgoing)
      return true;

    // Hide decay vertices that no longer have visible daughters.
    if (hasOutgoing && !hasVisibleOutgoing)
      return true;

    // Hide production/source vertices that no longer have visible outgoing particles.
    if (!hasIncoming && hasOutgoing && !hasVisibleOutgoing)
      return true;

    return false;
  }

  std::string appendEventIdToFilename(std::string const& filename, edm::EventID const& id) {
    const auto dotPos = filename.rfind('.');

    std::ostringstream ss;

    if (dotPos == std::string::npos) {
      ss << filename;
      ss << "_run" << id.run();
      ss << "_lumi" << id.luminosityBlock();
      ss << "_event" << id.event();
      return ss.str();
    }

    ss << filename.substr(0, dotPos);
    ss << "_run" << id.run();
    ss << "_lumi" << id.luminosityBlock();
    ss << "_event" << id.event();
    ss << filename.substr(dotPos);

    return ss.str();
  }

}  // namespace

class TruthLogicalGraphDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthLogicalGraphDumper(const edm::ParameterSet& cfg)
      : token_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
        rawToken_(mayConsume<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
        hitIndexTag_(cfg.getParameter<edm::InputTag>("hitIndex")),
        hitIndexToken_(mayConsume<truth::LogicalGraphHitIndex>(hitIndexTag_)),
        useHitIndex_(!hitIndexTag_.label().empty()),
        dotFile_(cfg.getParameter<std::string>("dotFile")),
        maxParticles_(cfg.getParameter<unsigned>("maxParticles")),
        maxVertices_(cfg.getParameter<unsigned>("maxVertices")),
        maxEdgesPerNode_(cfg.getParameter<unsigned>("maxEdgesPerNode")),
        hideLargeSimSourceVertices_(cfg.getParameter<bool>("hideLargeSimSourceVertices")),
        dumpSimHits_(cfg.getParameter<bool>("dumpSimHits")),
        largeSimSourceVertexMinOutgoing_(cfg.getParameter<unsigned>("largeSimSourceVertexMinOutgoing")),
        hideZeroSimHitSubgraphs_(cfg.getParameter<bool>("hideZeroSimHitSubgraphs")) {
    const auto hgcalRecHitTags = cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits");
    hgcalRecHitTags_.reserve(hgcalRecHitTags.size());
    hgcalRecHitTokens_.reserve(hgcalRecHitTags.size());

    for (auto const& tag : hgcalRecHitTags) {
      hgcalRecHitTags_.push_back(tag);
      hgcalRecHitTokens_.push_back(mayConsume<HGCRecHitCollection>(tag));
    }

    const auto pfRecHitTags = cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits");
    pfRecHitTags_.reserve(pfRecHitTags.size());
    pfRecHitTokens_.reserve(pfRecHitTags.size());

    for (auto const& tag : pfRecHitTags) {
      pfRecHitTags_.push_back(tag);
      pfRecHitTokens_.push_back(mayConsume<reco::PFRecHitCollection>(tag));
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"))
        ->setComment("Optional raw TruthGraph used only to enrich labels");

    desc.add<edm::InputTag>("hitIndex", edm::InputTag(""))
        ->setComment("Optional LogicalGraphHitIndex used to annotate particles with SimHit and RecHit summaries");

    // These two lists MUST match DetIdToRecHitMapProducer's hgcalRecHits/pfRecHits
    // (same collections, same order): hit.recHitIndex is the global concatenation
    // index into HGC-then-PF, so any divergence makes recHitEnergies[recHitIndex]
    // read the wrong hit. Defaults mirror that producer's defaults.
    desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")})
        ->setComment("HGCRecHit collections, in the same order used by DetIdToRecHitMapProducer");

    desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                         {edm::InputTag("particleFlowRecHitECAL", "Cleaned"),
                                          edm::InputTag("particleFlowRecHitHBHE", "Cleaned"),
                                          edm::InputTag("particleFlowRecHitHF", "Cleaned"),
                                          edm::InputTag("particleFlowRecHitHO", "Cleaned")})
        ->setComment("PFRecHit collections, in the same order used by DetIdToRecHitMapProducer");

    desc.add<std::string>("dotFile", "truthlogicalgraph.dot");

    desc.add<unsigned>("maxParticles", 5000)->setComment("Truncate logical particle nodes");
    desc.add<unsigned>("maxVertices", 5000)->setComment("Truncate logical vertex nodes");
    desc.add<unsigned>("maxEdgesPerNode", 200)->setComment("Truncate fanout per node");

    desc.add<bool>("hideLargeSimSourceVertices", true)
        ->setComment("If true, do not print large SIM-only source vertices in the DOT output");
    desc.add<bool>("dumpSimHits", false)->setComment("If true, dump all simhits");

    desc.add<unsigned>("largeSimSourceVertexMinOutgoing", 50)
        ->setComment("Minimum outgoing multiplicity for hiding a SIM-only source vertex");

    desc.add<bool>("hideZeroSimHitSubgraphs", false)
        ->setComment(
            "If true, hide every SIM-backed particle whose subgraph has zero SimHits, together with its descendant "
            "subgraph. Requires hitIndex to be configured.");

    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(const edm::Event& evt, const edm::EventSetup&) override {
    auto const& g = evt.get(token_);

    edm::Handle<TruthGraph> hRaw;
    evt.getByToken(rawToken_, hRaw);
    TruthGraph const* raw = hRaw.isValid() ? &(*hRaw) : nullptr;

    edm::Handle<truth::LogicalGraphHitIndex> hHitIndex;
    if (useHitIndex_) {
      evt.getByToken(hitIndexToken_, hHitIndex);
    }
    truth::LogicalGraphHitIndex const* hitIndex = hHitIndex.isValid() ? &(*hHitIndex) : nullptr;

    const std::vector<float> recHitEnergies = collectRecHitEnergies(evt);

    const std::string eventDotFile = appendEventIdToFilename(dotFile_, evt.id());

    std::ofstream os(eventDotFile);

    os << "digraph TruthLogicalGraph {\n";
    os << "  rankdir=LR;\n";
    os << "  node [fontsize=10];\n";

    const uint32_t nParticles = std::min<uint32_t>(g.nParticles(), maxParticles_);
    const uint32_t nVertices = std::min<uint32_t>(g.nVertices(), maxVertices_);

    std::vector<uint8_t> hideVertex(nVertices, 0);

    if (hideLargeSimSourceVertices_) {
      for (uint32_t i = 0; i < nVertices; ++i) {
        auto v = g.vertex(i);
        auto const& d = v.data();

        const auto incoming = v.incomingParticles();
        const auto outgoing = v.outgoingParticles();

        if (!d.hasGen() && d.hasSim() && incoming.empty() && outgoing.size() >= largeSimSourceVertexMinOutgoing_) {
          hideVertex[i] = 1;
        }
      }
    }

    std::vector<uint8_t> hideParticle(nParticles, 0);

    if (hideZeroSimHitSubgraphs_) {
      if (hitIndex == nullptr) {
        edm::LogWarning("TruthLogicalGraphDumper")
            << "hideZeroSimHitSubgraphs is enabled, but no valid LogicalGraphHitIndex was provided. "
            << "No zero-hit subgraphs will be hidden.";
      } else {
        for (uint32_t i = 0; i < nParticles; ++i) {
          if (!g.particle(i).data().hasSim())
            continue;

          if (i >= hitIndex->nParticles())
            continue;

          // "Hitless" means no detectable hit in ANY channel: a particle that
          // deposits only in the tracker / MTD / muon system (but never reaches
          // HGCal) must not be treated as hitless and hidden.
          bool hasAnySubgraphHit = false;
          for (std::size_t ch = 0; ch < truth::kNumHitChannels; ++ch) {
            if (!hitIndex->subgraphHits(static_cast<truth::HitChannel>(ch), i).empty()) {
              hasAnySubgraphHit = true;
              break;
            }
          }
          if (hasAnySubgraphHit)
            continue;

          hideParticle[i] = 1;

          forEachDescendantParticle(g, i, [&](uint32_t childId) {
            if (childId < hideParticle.size())
              hideParticle[childId] = 1;
          });
        }
      }
    }

    for (uint32_t i = 0; i < nVertices; ++i) {
      if (hideVertex[i])
        continue;

      if (shouldHideVertexAfterParticleFiltering(g, i, hideParticle)) {
        hideVertex[i] = 1;
      }
    }

    // Suppress particles that would render with no edge at all (floating "Particle
    // N" boxes): the leaf tail of a vertex whose outgoing edges are truncated past
    // maxEdgesPerNode_, or particles whose only incident vertices are hidden. They
    // are not graph orphans (their production vertex exists) - the crowded DOT view
    // just cannot draw the edge. Mirror the emission loops' per-node cap so the same
    // edges are considered drawn. (No-op in --showAll: maxEdgesPerNode_ is huge.)
    {
      std::vector<uint8_t> hasVisibleEdge(nParticles, 0);
      for (uint32_t v = 0; v < nVertices; ++v) {
        if (hideVertex[v])
          continue;
        unsigned kept = 0;
        for (uint32_t p : g.outgoingParticles(v)) {
          if (p >= nParticles || hideParticle[p])
            continue;
          hasVisibleEdge[p] = 1;  // a production edge v -> p will be drawn
          if (++kept >= maxEdgesPerNode_)
            break;
        }
      }
      for (uint32_t i = 0; i < nParticles; ++i) {
        if (hideParticle[i] || hasVisibleEdge[i])
          continue;
        for (uint32_t v : g.decayVertices(i)) {  // any visible decay vertex gives a drawn p -> v edge
          if (v < nVertices && !hideVertex[v]) {
            hasVisibleEdge[i] = 1;
            break;
          }
        }
      }
      for (uint32_t i = 0; i < nParticles; ++i) {
        if (!hideParticle[i] && !hasVisibleEdge[i])
          hideParticle[i] = 1;
      }
    }

    // ------------------------------------------------------------------
    // Particle nodes
    // ------------------------------------------------------------------
    for (uint32_t i = 0; i < nParticles; ++i) {
      if (hideParticle[i])
        continue;

      auto p = g.particle(i);
      auto const& d = p.data();

      const bool hasHitInfo = hitIndex != nullptr && i < hitIndex->nParticles();

      const auto directHits = hasHitInfo ? hitIndex->directHits(truth::HitChannel::HGCalCalo, i)
                                         : std::span<const truth::LogicalGraphHitIndex::Hit>();
      const auto subgraphHits = hasHitInfo ? hitIndex->subgraphHits(truth::HitChannel::HGCalCalo, i)
                                           : std::span<const truth::LogicalGraphHitIndex::Hit>();

      const HitSummary directSummary = hasHitInfo ? summarizeHits(directHits, recHitEnergies) : HitSummary();
      const HitSummary subgraphSummary = hasHitInfo ? summarizeHits(subgraphHits, recHitEnergies) : HitSummary();

      // Tracker simhits (separate channel, no recHit association). Reusing
      // summarizeHits is fine: tracker hits have an invalid recHitIndex, so only
      // nSimHits and simHitEnergy (energy loss) carry meaning.
      const bool hasTrackerInfo = hasHitInfo && hitIndex->hasChannel(truth::HitChannel::Tracker);
      const auto trackerDirectHits = hasTrackerInfo ? hitIndex->directHits(truth::HitChannel::Tracker, i)
                                                    : std::span<const truth::LogicalGraphHitIndex::Hit>();
      const auto trackerSubgraphHits = hasTrackerInfo ? hitIndex->subgraphHits(truth::HitChannel::Tracker, i)
                                                      : std::span<const truth::LogicalGraphHitIndex::Hit>();
      const HitSummary trackerDirectSummary =
          hasTrackerInfo ? summarizeHits(trackerDirectHits, recHitEnergies) : HitSummary();
      const HitSummary trackerSubgraphSummary =
          hasTrackerInfo ? summarizeHits(trackerSubgraphHits, recHitEnergies) : HitSummary();

      // MTD (BTL/ETL) simhits from the MtdSimLayerCluster channel.
      const bool hasMtdInfo = hasHitInfo && hitIndex->hasChannel(truth::HitChannel::MTD);
      const HitSummary mtdDirectSummary =
          hasMtdInfo ? summarizeHits(hitIndex->directHits(truth::HitChannel::MTD, i), recHitEnergies) : HitSummary();
      const HitSummary mtdSubgraphSummary =
          hasMtdInfo ? summarizeHits(hitIndex->subgraphHits(truth::HitChannel::MTD, i), recHitEnergies) : HitSummary();

      // MTD recHitIndex points into the FTLCluster ordering (channel-relative, not
      // the HGCal recHit ordering), so count the FTLCluster-linked hits directly.
      uint32_t mtdSubgraphRecHitLinked = 0;
      if (hasMtdInfo)
        for (auto const& h : hitIndex->subgraphHits(truth::HitChannel::MTD, i))
          mtdSubgraphRecHitLinked += static_cast<uint32_t>(h.hasRecHit());

      // Muon-chamber simhits (DT/CSC/RPC/GEM), no recHit link.
      const bool hasMuonInfo = hasHitInfo && hitIndex->hasChannel(truth::HitChannel::Muon);
      const HitSummary muonDirectSummary =
          hasMuonInfo ? summarizeHits(hitIndex->directHits(truth::HitChannel::Muon, i), recHitEnergies) : HitSummary();
      const HitSummary muonSubgraphSummary =
          hasMuonInfo ? summarizeHits(hitIndex->subgraphHits(truth::HitChannel::Muon, i), recHitEnergies)
                      : HitSummary();

      os << "  p" << i << " [shape=ellipse, hasCheckpoints=" << p.hasCheckpoints() << ", hasGen=" << p.hasGen()
         << ", hasSim=" << d.hasSim();

      if (p.hasCheckpoints()) {
        os << ", color=\"red\", penwidth=2";
      } else if (d.hasGen() && d.hasSim()) {
        os << ", penwidth=2";
      } else if (d.hasGen()) {
        os << ", color=\"blue\"";
      } else if (d.hasSim()) {
        os << ", color=\"darkgreen\"";
      }

      os << ", pid=" << d.pdgId << ", status=" << d.status << ", statusFlags=" << d.statusFlags << ", flags=<"
         << statusFlagsLabel(d.statusFlags) << ">"
         << ", eid=" << d.eventId << ", genEvent=" << d.genEvent << ", isRoot=" << p.isRoot()
         << ", isLeaf=" << p.isLeaf() << ", p4=\"" << fmtP4(d.momentum)
         << "\", nProdVtx=" << p.productionVertices().size() << ", nDecayVtx=" << p.decayVertices().size()
         << ", nParents=" << p.parents().size() << ", nChildren=" << p.children().size()
         << ", nCheckpoints=" << d.checkpoints.size() << ", backscattered=" << d.backscattered;

      if (hasHitInfo) {
        os << ", nDirectSimHits=" << directSummary.nSimHits << ", nDirectRecHits=" << directSummary.nMatchedRecHits
           << ", directSimHitEnergy=" << fmtEnergy(directSummary.simHitEnergy)
           << ", directRecHitEnergy=" << fmtEnergy(directSummary.recHitEnergy)
           << ", nSubgraphSimHits=" << subgraphSummary.nSimHits
           << ", nSubgraphRecHits=" << subgraphSummary.nMatchedRecHits
           << ", subgraphSimHitEnergy=" << fmtEnergy(subgraphSummary.simHitEnergy)
           << ", subgraphRecHitEnergy=" << fmtEnergy(subgraphSummary.recHitEnergy);
        if (hasTrackerInfo) {
          os << ", nDirectTrackerSimHits=" << trackerDirectSummary.nSimHits
             << ", directTrackerSimHitEnergy=" << fmtEnergy(trackerDirectSummary.simHitEnergy)
             << ", nSubgraphTrackerSimHits=" << trackerSubgraphSummary.nSimHits
             << ", subgraphTrackerSimHitEnergy=" << fmtEnergy(trackerSubgraphSummary.simHitEnergy);
        }
        if (hasMtdInfo) {
          os << ", nDirectMtdSimHits=" << mtdDirectSummary.nSimHits
             << ", directMtdSimHitEnergy=" << fmtEnergy(mtdDirectSummary.simHitEnergy)
             << ", nSubgraphMtdSimHits=" << mtdSubgraphSummary.nSimHits
             << ", subgraphMtdSimHitEnergy=" << fmtEnergy(mtdSubgraphSummary.simHitEnergy)
             << ", nSubgraphMtdRecHits=" << mtdSubgraphRecHitLinked;
        }
        if (hasMuonInfo) {
          os << ", nDirectMuonSimHits=" << muonDirectSummary.nSimHits
             << ", directMuonSimHitEnergy=" << fmtEnergy(muonDirectSummary.simHitEnergy)
             << ", nSubgraphMuonSimHits=" << muonSubgraphSummary.nSimHits
             << ", subgraphMuonSimHitEnergy=" << fmtEnergy(muonSubgraphSummary.simHitEnergy);
        }
        if (dumpSimHits_) {
          os << ", directHitsDetIds=\"";
          for (auto h : directHits) {
            os << h.detId << ",";
          }
          os << "\"";
          os << ", directHitsEnergies=\"";
          for (auto h : directHits) {
            os << h.energy << ",";
          }
          os << "\"";
        }
      }

      if (raw != nullptr) {
        os << ", raw_GEN=<" << rawNodeSummary(raw, d.genNode) << ">, raw_SIM=<" << rawNodeSummary(raw, d.simNode)
           << ">";
      }

      // Big, immediately-legible particle name + PDG id as the table's title row
      // (HTML-like labels cannot mix free text and a TABLE), with the details below.
      const std::string bigName = (!p.hasGen() && !d.hasSim()) ? std::string("connector") : pdgLabel(d.pdgId);
      os << ", label=<\n";
      os << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
      os << "      <TR><TD><FONT POINT-SIZE=\"22\"><B>" << bigName << "</B></FONT></TD></TR>\n";
      os << "      <TR><TD><B>Particle " << i << "</B></TD></TR>\n";

      if (d.pdgId != 0)
        os << "      <TR><TD>pid: " << pdgLabel(d.pdgId) << "</TD></TR>\n";

      if (d.status != 0)
        os << "      <TR><TD>status: " << d.status << "</TD></TR>\n";

      if (d.statusFlags != 0) {
        os << "      <TR><TD>statusFlags: " << d.statusFlags << "</TD></TR>\n";
        os << "      <TR><TD>flags: " << statusFlagsLabel(d.statusFlags) << "</TD></TR>\n";
      }

      if (d.backscattered)
        os << "      <TR><TD><B>back-scattered</B></TD></TR>\n";

      if (d.eventId != 0)
        os << "      <TR><TD>eid: " << d.eventId << "</TD></TR>\n";

      if (d.genEvent >= 0)
        os << "      <TR><TD>genEvent: " << d.genEvent << "</TD></TR>\n";

      os << "      <TR><TD>hasGen: " << (d.hasGen() ? "yes" : "no") << "  hasSim: " << (d.hasSim() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>isRoot: " << (p.isRoot() ? "yes" : "no") << "  isLeaf: " << (p.isLeaf() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>p4: " << fmtP4(d.momentum) << "</TD></TR>\n";

      os << "      <TR><TD>nProdVtx: " << p.productionVertices().size() << "  nDecayVtx: " << p.decayVertices().size()
         << "</TD></TR>\n";

      os << "      <TR><TD>nParents: " << p.parents().size() << "  nChildren: " << p.children().size()
         << "</TD></TR>\n";

      os << "      <TR><TD>nCheckpoints: " << d.checkpoints.size() << "</TD></TR>\n";

      if (hasHitInfo) {
        os << "      <TR><TD><B>direct simHits:</B> " << directSummary.nSimHits
           << "  simE=" << fmtEnergy(directSummary.simHitEnergy) << "</TD></TR>\n";
        os << "      <TR><TD><B>direct recHits:</B> " << directSummary.nMatchedRecHits
           << "  missing=" << directSummary.nMissingRecHits << "  recoE=" << fmtEnergy(directSummary.recHitEnergy)
           << "</TD></TR>\n";

        os << "      <TR><TD><B>subgraph simHits:</B> " << subgraphSummary.nSimHits
           << "  simE=" << fmtEnergy(subgraphSummary.simHitEnergy) << "</TD></TR>\n";
        os << "      <TR><TD><B>subgraph recHits:</B> " << subgraphSummary.nMatchedRecHits
           << "  missing=" << subgraphSummary.nMissingRecHits << "  recoE=" << fmtEnergy(subgraphSummary.recHitEnergy)
           << "</TD></TR>\n";
      }

      if (hasTrackerInfo) {
        os << "      <TR><TD><B>direct tracker simHits:</B> " << trackerDirectSummary.nSimHits
           << "  dE=" << fmtEnergy(trackerDirectSummary.simHitEnergy) << "</TD></TR>\n";
        os << "      <TR><TD><B>subgraph tracker simHits:</B> " << trackerSubgraphSummary.nSimHits
           << "  dE=" << fmtEnergy(trackerSubgraphSummary.simHitEnergy) << "</TD></TR>\n";
      }

      for (auto const& cp : d.checkpoints) {
        os << "      <TR><TD><FONT COLOR=\"red\">checkpointId: " << cp.checkpointId << "</FONT></TD></TR>\n";
        os << "      <TR><TD><FONT COLOR=\"red\">x4@checkpoint: " << fmtP4(cp.position) << "</FONT></TD></TR>\n";
        os << "      <TR><TD><FONT COLOR=\"red\">p4@checkpoint: " << fmtP4(cp.momentum) << "</FONT></TD></TR>\n";
      }

      if (raw != nullptr) {
        os << "      <TR><TD>raw GEN: " << rawNodeSummary(raw, d.genNode) << "</TD></TR>\n";
        os << "      <TR><TD>raw SIM: " << rawNodeSummary(raw, d.simNode) << "</TD></TR>\n";
      }

      os << "    </TABLE>\n";
      os << "  >];\n";
    }

    // ------------------------------------------------------------------
    // Vertex nodes
    // ------------------------------------------------------------------
    for (uint32_t i = 0; i < nVertices; ++i) {
      if (hideVertex[i])
        continue;

      auto v = g.vertex(i);
      auto const& d = v.data();

      const auto& incoming = v.incomingParticles();
      const auto& outgoing = v.outgoingParticles();

      const char* roleName = nullptr;
      const char* roleColor = "lightgrey";
      switch (d.vertexRole()) {
        case truth::VertexRole::Interaction:
          roleName = "interaction";
          roleColor = "indianred1";
          break;
        case truth::VertexRole::Upstream:
          roleName = "ISR/upstream";
          roleColor = "navajowhite";
          break;
        case truth::VertexRole::UnderlyingEvent:
          roleName = "underlying event";
          roleColor = "lightgrey";
          break;
        default:
          break;
      }

      os << "  v" << i << " [shape=diamond, domain=<" << logicalVertexDomain(d) << ">, hasGen=" << d.hasGen()
         << ", hasSim=" << d.hasSim() << ", eid=" << d.eventId << ", genEvent=" << d.genEvent << ", reason=\""
         << truth::vertexReasonName(d.vertexReason()) << "\"" << ", isSource=" << v.isSource()
         << ", isSink=" << v.isSink();
      if (roleName != nullptr) {
        os << ", role=\"" << roleName << "\", style=filled, fillcolor=\"" << roleColor << "\"";
      } else if (d.hasGen() && d.hasSim()) {
        os << ", color=\"purple\", penwidth=2";
      } else if (d.hasGen()) {
        os << ", color=\"blue\"";
      } else if (d.hasSim()) {
        os << ", color=\"darkgreen\"";
      }
      os << ", x=" << std::fixed << std::setprecision(6) << d.position.x() << ", y=" << d.position.y()
         << ", z=" << d.position.z() << ", t=" << d.position.t() << ", x4=\"" << fmtX4(d.position) << "\""
         << ", nIn=" << incoming.size() << ", nOut=" << outgoing.size();

      if (raw != nullptr) {
        os << ", raw_GEN=<" << rawNodeSummary(raw, d.genNode) << ">, raw_SIM=<" << rawNodeSummary(raw, d.simNode)
           << ">";
      }

      // Big title row: the role for artificial vertices; the physical reason when
      // known (SIM vertices carry the Geant4 process); else - GEN-only vertices have
      // no process - infer from the topology (a single incoming particle decays).
      std::string vBig;
      if (roleName != nullptr)
        vBig = roleName;
      else if (d.vertexReason() != truth::VertexReason::Unknown)
        vBig = truth::vertexReasonName(d.vertexReason());
      else
        vBig = (incoming.size() == 1) ? "decay" : "production";
      os << ", label=<\n";
      os << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
      os << "      <TR><TD><FONT POINT-SIZE=\"18\"><B>" << vBig << "</B></FONT></TD></TR>\n";
      os << "      <TR><TD><B>Vertex " << i << "</B></TD></TR>\n";

      if (roleName != nullptr)
        os << "      <TR><TD><B>" << roleName << "</B> (genEvent=" << d.genEvent << ", eid=" << d.eventId
           << ")</TD></TR>\n";

      os << "      <TR><TD>domain: " << logicalVertexDomain(d) << "</TD></TR>\n";

      if (d.vertexReason() != truth::VertexReason::Unknown)
        os << "      <TR><TD>reason: " << truth::vertexReasonName(d.vertexReason()) << "</TD></TR>\n";

      if (d.eventId != 0)
        os << "      <TR><TD>eid: " << d.eventId << "</TD></TR>\n";

      if (d.genEvent >= 0)
        os << "      <TR><TD>genEvent: " << d.genEvent << "</TD></TR>\n";

      os << "      <TR><TD>hasGen: " << (d.hasGen() ? "yes" : "no") << "  hasSim: " << (d.hasSim() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>isSource: " << (v.isSource() ? "yes" : "no") << "  isSink: " << (v.isSink() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>x4: " << fmtX4(d.position) << "</TD></TR>\n";

      os << "      <TR><TD>nIn: " << v.incomingParticles().size() << "  nOut: " << v.outgoingParticles().size()
         << "</TD></TR>\n";

      if (raw != nullptr) {
        os << "      <TR><TD>raw GEN: " << rawNodeSummary(raw, d.genNode) << "</TD></TR>\n";
        os << "      <TR><TD>raw SIM: " << rawNodeSummary(raw, d.simNode) << "</TD></TR>\n";
      }

      os << "    </TABLE>\n";
      os << "  >];\n";
    }

    // ------------------------------------------------------------------
    // Edges: physics-forward only
    // ------------------------------------------------------------------
    for (uint32_t i = 0; i < nParticles; ++i) {
      if (hideParticle[i])
        continue;

      unsigned kept = 0;

      for (uint32_t v : g.decayVertices(i)) {
        if (v >= nVertices)
          continue;

        if (hideVertex[v])
          continue;

        os << "  p" << i << " -> v" << v << ";\n";

        if (++kept >= maxEdgesPerNode_)
          break;
      }
    }

    for (uint32_t i = 0; i < nVertices; ++i) {
      if (hideVertex[i])
        continue;

      unsigned kept = 0;

      for (uint32_t p : g.outgoingParticles(i)) {
        if (p >= nParticles)
          continue;

        if (hideParticle[p])
          continue;

        os << "  v" << i << " -> p" << p << ";\n";

        if (++kept >= maxEdgesPerNode_)
          break;
      }
    }

    os << "}\n";
  }

private:
  std::vector<float> collectRecHitEnergies(const edm::Event& evt) const {
    std::vector<float> energies;

    // This must match the global recHit indexing order used by DetIdToRecHitMapProducer:
    // first all HGCRecHit collections, then all PFRecHit collections.
    for (uint32_t i = 0; i < hgcalRecHitTokens_.size(); ++i) {
      edm::Handle<HGCRecHitCollection> handle;
      evt.getByToken(hgcalRecHitTokens_[i], handle);

      if (!handle.isValid()) {
        edm::LogWarning("TruthLogicalGraphDumper") << "Missing HGCRecHit collection " << hgcalRecHitTags_[i].encode()
                                                   << ". Skipping it while rebuilding recHit energies.";
        continue;
      }

      energies.reserve(energies.size() + handle->size());
      for (auto const& hit : *handle) {
        energies.push_back(hit.energy());
      }
    }
    for (uint32_t i = 0; i < pfRecHitTokens_.size(); ++i) {
      edm::Handle<reco::PFRecHitCollection> handle;
      evt.getByToken(pfRecHitTokens_[i], handle);

      if (!handle.isValid()) {
        edm::LogWarning("TruthLogicalGraphDumper") << "Missing reco::PFRecHitCollection " << pfRecHitTags_[i].encode()
                                                   << ". Skipping it while rebuilding recHit energies.";
        continue;
      }

      energies.reserve(energies.size() + handle->size());
      for (auto const& hit : *handle) {
        energies.push_back(hit.energy());
      }
    }

    return energies;
  }

  edm::EDGetTokenT<truth::Graph> token_;
  edm::EDGetTokenT<TruthGraph> rawToken_;

  edm::InputTag hitIndexTag_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  bool useHitIndex_;

  std::vector<edm::InputTag> hgcalRecHitTags_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;

  std::vector<edm::InputTag> pfRecHitTags_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;

  std::string dotFile_;
  unsigned maxParticles_;
  unsigned maxVertices_;
  unsigned maxEdgesPerNode_;
  bool hideLargeSimSourceVertices_;
  bool dumpSimHits_;
  unsigned largeSimSourceVertexMinOutgoing_;
  bool hideZeroSimHitSubgraphs_;
};

DEFINE_FWK_MODULE(TruthLogicalGraphDumper);
