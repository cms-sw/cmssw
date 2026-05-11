#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {

  std::string pdgNameUtf8(int pdgId) {
    const int ap = std::abs(pdgId);

    if (pdgId == 11)
      return "e⁻";
    if (pdgId == -11)
      return "e⁺";
    if (pdgId == 13)
      return "μ⁻";
    if (pdgId == -13)
      return "μ⁺";
    if (pdgId == 15)
      return "τ⁻";
    if (pdgId == -15)
      return "τ⁺";

    if (pdgId == 12)
      return "νₑ";
    if (pdgId == -12)
      return "ν̄ₑ";
    if (pdgId == 14)
      return "ν_μ";
    if (pdgId == -14)
      return "ν̄_μ";
    if (pdgId == 16)
      return "ν_τ";
    if (pdgId == -16)
      return "ν̄_τ";

    if (pdgId == 22)
      return "γ";
    if (pdgId == 21)
      return "g";
    if (pdgId == 23)
      return "Z⁰";
    if (pdgId == 24)
      return "W⁺";
    if (pdgId == -24)
      return "W⁻";
    if (pdgId == 25)
      return "H";

    if (pdgId == 2212)
      return "p";
    if (pdgId == -2212)
      return "p̄";
    if (pdgId == 2112)
      return "n";
    if (pdgId == -2112)
      return "n̄";

    if (pdgId == 111)
      return "π⁰";
    if (pdgId == 211)
      return "π⁺";
    if (pdgId == -211)
      return "π⁻";
    if (pdgId == 321)
      return "K⁺";
    if (pdgId == -321)
      return "K⁻";
    if (pdgId == 130)
      return "K⁰_L";
    if (pdgId == 310)
      return "K⁰_S";

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

  template <typename P4>
  std::string fmtP4(P4 const& p4) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << "(" << p4.px() << ", " << p4.py() << ", " << p4.pz() << ", " << p4.e() << ")";
    return ss.str();
  }

  const char* logicalVertexDomain(truth::VertexData const& d) {
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
}  // namespace

class TruthLogicalGraphDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthLogicalGraphDumper(const edm::ParameterSet& cfg)
      : token_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
        rawToken_(mayConsume<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
        dotFile_(cfg.getParameter<std::string>("dotFile")),
        maxParticles_(cfg.getParameter<unsigned>("maxParticles")),
        maxVertices_(cfg.getParameter<unsigned>("maxVertices")),
        maxEdgesPerNode_(cfg.getParameter<unsigned>("maxEdgesPerNode")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"))
        ->setComment("Optional raw TruthGraph used only to enrich labels");
    desc.add<std::string>("dotFile", "truthlogicalgraph.dot");
    desc.add<unsigned>("maxParticles", 5000)->setComment("Truncate logical particle nodes");
    desc.add<unsigned>("maxVertices", 5000)->setComment("Truncate logical vertex nodes");
    desc.add<unsigned>("maxEdgesPerNode", 200)->setComment("Truncate fanout per node");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(const edm::Event& evt, const edm::EventSetup&) override {
    auto const& g = evt.get(token_);

    edm::Handle<TruthGraph> hRaw;
    evt.getByToken(rawToken_, hRaw);
    TruthGraph const* raw = hRaw.isValid() ? &(*hRaw) : nullptr;

    std::ofstream os(dotFile_);
    os << "digraph TruthLogicalGraph {\n";
    os << "  rankdir=LR;\n";
    os << "  node [fontsize=10];\n";

    const uint32_t nParticles = std::min<uint32_t>(g.nParticles(), maxParticles_);
    const uint32_t nVertices = std::min<uint32_t>(g.nVertices(), maxVertices_);

    // ------------------------------------------------------------------
    // Particle nodes
    // ------------------------------------------------------------------
    for (uint32_t i = 0; i < nParticles; ++i) {
      auto p = g.particle(i);
      auto const& d = p.data();

      os << "  p" << i << " [shape=ellipse";

      if (p.hasCheckpoints()) {
        os << ", color=\"red\", penwidth=2";
      } else if (d.hasGen() && d.hasSim()) {
        os << ", penwidth=2";
      } else if (d.hasGen()) {
        os << ", color=\"blue\"";
      } else if (d.hasSim()) {
        os << ", color=\"darkgreen\"";
      }

      os << ", label=<\n";
      os << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
      os << "      <TR><TD><B>Particle " << i << "</B></TD></TR>\n";

      if (d.pdgId != 0)
        os << "      <TR><TD>pid: " << pdgLabel(d.pdgId) << "</TD></TR>\n";
      if (d.status != 0)
        os << "      <TR><TD>status: " << d.status << "</TD></TR>\n";
      if (d.statusFlags != 0) {
        os << "      <TR><TD>statusFlags: " << d.statusFlags << "</TD></TR>\n";
        os << "      <TR><TD>flags: " << statusFlagsLabel(d.statusFlags) << "</TD></TR>\n";
      }
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
      auto v = g.vertex(i);
      auto const& d = v.data();

      os << "  v" << i << " [shape=diamond";

      if (d.hasGen() && d.hasSim()) {
        os << ", color=\"purple\", penwidth=2";
      } else if (d.hasGen()) {
        os << ", color=\"blue\"";
      } else if (d.hasSim()) {
        os << ", color=\"darkgreen\"";
      }

      os << ", label=<\n";
      os << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
      os << "      <TR><TD><B>Vertex " << i << "</B></TD></TR>\n";

      os << "      <TR><TD>domain: " << logicalVertexDomain(d) << "</TD></TR>\n";

      if (d.eventId != 0)
        os << "      <TR><TD>eid: " << d.eventId << "</TD></TR>\n";
      if (d.genEvent >= 0)
        os << "      <TR><TD>genEvent: " << d.genEvent << "</TD></TR>\n";

      os << "      <TR><TD>hasGen: " << (d.hasGen() ? "yes" : "no") << "  hasSim: " << (d.hasSim() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>isSource: " << (v.isSource() ? "yes" : "no") << "  isSink: " << (v.isSink() ? "yes" : "no")
         << "</TD></TR>\n";

      os << "      <TR><TD>x4: " << fmtP4(d.position) << "</TD></TR>\n";

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
      unsigned kept = 0;
      for (uint32_t v : g.decayVertices(i)) {
        if (v >= nVertices)
          continue;
        os << "  p" << i << " -> v" << v << ";\n";
        if (++kept >= maxEdgesPerNode_)
          break;
      }
    }

    for (uint32_t i = 0; i < nVertices; ++i) {
      unsigned kept = 0;
      for (uint32_t p : g.outgoingParticles(i)) {
        if (p >= nParticles)
          continue;
        os << "  v" << i << " -> p" << p << ";\n";
        if (++kept >= maxEdgesPerNode_)
          break;
      }
    }

    os << "}\n";
    os.close();
  }

private:
  edm::EDGetTokenT<truth::Graph> token_;
  edm::EDGetTokenT<TruthGraph> rawToken_;

  std::string dotFile_;
  unsigned maxParticles_;
  unsigned maxVertices_;
  unsigned maxEdgesPerNode_;
};

DEFINE_FWK_MODULE(TruthLogicalGraphDumper);
