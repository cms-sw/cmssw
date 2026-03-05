#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {

  // --- PDG naming (UTF-8)
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

  template <typename P4T>
  std::string fmtP4(const P4T& p4) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(3) << "(" << p4.px() << ", " << p4.py() << ", " << p4.pz() << ", " << p4.e() << ")";
    return ss.str();
  }

  template <typename X4T>
  std::string fmtX4(const X4T& x4) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(3) << "(" << x4.x() << ", " << x4.y() << ", " << x4.z() << ", " << x4.t() << ")";
    return ss.str();
  }

  const char* kindName(TruthGraph::NodeKind k) {
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

  const char* shapeFor(TruthGraph::NodeKind k) {
    switch (k) {
      case TruthGraph::NodeKind::GenEvent:
        return "box";
      case TruthGraph::NodeKind::GenVertex:
        return "diamond";
      case TruthGraph::NodeKind::GenParticle:
        return "box";
      case TruthGraph::NodeKind::SimVertex:
        return "diamond";
      case TruthGraph::NodeKind::SimTrack:
        return "ellipse";
    }
    return "box";
  }

  const char* edgeAttrs(uint8_t ek) {
    using EK = TruthGraph::EdgeKind;
    switch (static_cast<EK>(ek)) {
      case EK::Gen:
        return "";
      case EK::Sim:
        return " [style=solid]";
      case EK::GenToSim:
        return " [dir=both, style=dashed]";
      case EK::SimToGen:
        return " [dir=both, style=dotted]";
    }
    return "";
  }

}  // anonymous namespace

class TruthGraphDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthGraphDumper(const edm::ParameterSet& cfg)
      : token_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("src"))),
        dotFile_(cfg.getParameter<std::string>("dotFile")),
        maxNodes_(cfg.getParameter<unsigned>("maxNodes")),
        maxEdgesPerNode_(cfg.getParameter<unsigned>("maxEdgesPerNode")),
        simTracksToken_(mayConsume<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
        simVerticesToken_(mayConsume<edm::SimVertexContainer>(cfg.getParameter<edm::InputTag>("simVertices"))),
        hepmc2Token_(mayConsume<edm::HepMCProduct>(cfg.getParameter<edm::InputTag>("genEventHepMC"))),
        hepmc3Token_(mayConsume<edm::HepMC3Product>(cfg.getParameter<edm::InputTag>("genEventHepMC3"))) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("truthGraphProducer"));
    desc.add<std::string>("dotFile", "truthgraph.dot");
    desc.add<unsigned>("maxNodes", 5000)->setComment("Truncate to keep DOT manageable");
    desc.add<unsigned>("maxEdgesPerNode", 200)->setComment("Truncate fanout per node");

    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"))
        ->setComment("SimTrackContainer (optional, used to enrich SimTrack nodes)");
    desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"))
        ->setComment("SimVertexContainer (optional, used for future enrichment)");

    // GEN record (for enriching GenParticle/GenVertex nodes)
    desc.add<edm::InputTag>("genEventHepMC", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMCProduct label (your step1.root shows this is present)");
    desc.add<edm::InputTag>("genEventHepMC3", edm::InputTag("generatorSmeared"))
        ->setComment("edm::HepMC3Product label (optional)");

    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(const edm::Event& evt, const edm::EventSetup&) override {
    auto const& g = evt.get(token_);

    // --- SIM handles (optional)
    edm::Handle<edm::SimTrackContainer> hSimTracks;
    evt.getByToken(simTracksToken_, hSimTracks);

    std::unordered_map<uint32_t, uint32_t> tidToIndex;
    if (hSimTracks.isValid()) {
      tidToIndex.reserve(hSimTracks->size() * 2);
      for (uint32_t i = 0; i < hSimTracks->size(); ++i) {
        tidToIndex.emplace((*hSimTracks)[i].trackId(), i);
      }
    }

    // --- GEN handles (optional)
    // Prefer HepMC2 if present (it is in your step1.root); else HepMC3.
    edm::Handle<edm::HepMCProduct> hHepMC2;
    evt.getByToken(hepmc2Token_, hHepMC2);

    edm::Handle<edm::HepMC3Product> hHepMC3;
    evt.getByToken(hepmc3Token_, hHepMC3);

    // HepMC2 lookup maps
    std::unordered_map<int, HepMC::GenParticle const*> bc2p;
    std::unordered_map<int, HepMC::GenVertex const*> bc2v;
    HepMC::GenEvent const* ev2 = nullptr;

    if (hHepMC2.isValid() && hHepMC2->GetEvent() != nullptr) {
      ev2 = hHepMC2->GetEvent();
      bc2p.reserve(ev2->particles_size() * 2);
      bc2v.reserve(ev2->vertices_size() * 2);

      for (auto p = ev2->particles_begin(); p != ev2->particles_end(); ++p) {
        bc2p.emplace((*p)->barcode(), *p);
      }
      for (auto v = ev2->vertices_begin(); v != ev2->vertices_end(); ++v) {
        bc2v.emplace((*v)->barcode(), *v);
      }
    }

    // HepMC3 reconstruction + maps
    HepMC3::GenEvent ev3;
    std::unordered_map<int, HepMC3::ConstGenParticlePtr> id3p;
    std::unordered_map<int, HepMC3::ConstGenVertexPtr> id3v;
    bool have3 = false;

    if (!ev2 && hHepMC3.isValid() && hHepMC3->GetEvent() != nullptr) {
      have3 = true;
      const HepMC3::GenEventData* data = hHepMC3->GetEvent();
      ev3.read_data(*data);

      id3p.reserve(ev3.particles().size() * 2);
      id3v.reserve(ev3.vertices().size() * 2);

      for (auto const& pptr : ev3.particles()) {
        if (pptr)
          id3p.emplace(pptr->id(), pptr);
      }
      for (auto const& vptr : ev3.vertices()) {
        if (vptr)
          id3v.emplace(vptr->id(), vptr);
      }
    }

    std::ofstream os(dotFile_);
    os << "digraph TruthGraph {\n";
    os << "  rankdir=LR;\n";
    os << "  node [fontsize=10];\n";

    const uint32_t n = std::min<uint32_t>(g.nNodes(), maxNodes_);

    // nodes
    for (uint32_t i = 0; i < n; ++i) {
      auto const& r = g.nodeRef(i);
      const auto pdg = g.nodePdgId(i);
      const auto st = g.nodeStatus(i);
      const auto eid = g.nodeEventId(i);

      // SimTrack enrichment
      bool crossedBoundary = false;
      bool haveSim = false;
      SimTrack const* simt = nullptr;

      if (r.kind == TruthGraph::NodeKind::SimTrack && hSimTracks.isValid()) {
        const int64_t key = r.key;  // trackId
        if (key >= 0 && key <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
          auto it = tidToIndex.find(static_cast<uint32_t>(key));
          if (it != tidToIndex.end()) {
            simt = &(*hSimTracks)[it->second];
            haveSim = true;
            crossedBoundary = simt->crossedBoundary();
          }
        }
      }

      // Node style
      os << "  n" << i << " [shape=" << shapeFor(r.kind);
      if (crossedBoundary)
        os << ", color=\"red\", penwidth=2";

      // HTML label
      os << ", label=<\n";
      os << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
      os << "      <TR><TD><B>" << i << " " << kindName(r.kind) << "</B> key=" << r.key << "</TD></TR>\n";

      if (pdg != 0)
        os << "      <TR><TD>pid: " << pdgLabel(pdg) << "</TD></TR>\n";
      if (st != 0)
        os << "      <TR><TD>status: " << st << "</TD></TR>\n";
      if (eid != 0)
        os << "      <TR><TD>eid: " << eid << "</TD></TR>\n";

      // --- GEN enrichment
      if (r.kind == TruthGraph::NodeKind::GenEvent) {
        if (ev2) {
          os << "      <TR><TD>HepMC2: event=" << ev2->event_number() << " spid=" << ev2->signal_process_id()
             << "</TD></TR>\n";
        } else if (have3) {
          os << "      <TR><TD>HepMC3: event=" << ev3.event_number() << "</TD></TR>\n";
        }
      } else if (r.kind == TruthGraph::NodeKind::GenParticle) {
        const int bc = static_cast<int>(r.key);
        if (ev2) {
          auto it = bc2p.find(bc);
          if (it != bc2p.end()) {
            auto const* p = it->second;
            os << "      <TR><TD>pid: " << pdgLabel(p->pdg_id()) << "</TD></TR>\n";
            os << "      <TR><TD>status: " << p->status() << "</TD></TR>\n";
            os << "      <TR><TD>p4: " << fmtP4(p->momentum()) << "</TD></TR>\n";
            os << "      <TR><TD>m: " << std::fixed << std::setprecision(3) << p->generated_mass() << "</TD></TR>\n";
            const int prod = p->production_vertex() ? p->production_vertex()->barcode() : 0;
            const int endv = p->end_vertex() ? p->end_vertex()->barcode() : 0;
            os << "      <TR><TD>prodVtx: " << prod << " endVtx: " << endv << "</TD></TR>\n";
          }
        } else if (have3) {
          auto it = id3p.find(bc);
          if (it != id3p.end() && it->second) {
            auto const& p = it->second;
            os << "      <TR><TD>pid: " << pdgLabel(p->pid()) << "</TD></TR>\n";
            os << "      <TR><TD>status: " << p->status() << "</TD></TR>\n";
            os << "      <TR><TD>p4: " << fmtP4(p->momentum()) << "</TD></TR>\n";
            const int prod = p->production_vertex() ? p->production_vertex()->id() : 0;
            const int endv = p->end_vertex() ? p->end_vertex()->id() : 0;
            os << "      <TR><TD>prodVtx: " << prod << " endVtx: " << endv << "</TD></TR>\n";
          }
        }
      } else if (r.kind == TruthGraph::NodeKind::GenVertex) {
        const int bc = static_cast<int>(r.key);
        if (ev2) {
          auto it = bc2v.find(bc);
          if (it != bc2v.end()) {
            auto const* v = it->second;
            os << "      <TR><TD>barcode: " << v->barcode() << "</TD></TR>\n";
            os << "      <TR><TD>x4: " << fmtX4(v->position()) << "</TD></TR>\n";
            os << "      <TR><TD>nIn: " << v->particles_in_size() << " nOut: " << v->particles_out_size()
               << "</TD></TR>\n";
          }
        } else if (have3) {
          auto it = id3v.find(bc);
          if (it != id3v.end() && it->second) {
            auto const& v = it->second;
            os << "      <TR><TD>status: " << v->status() << "</TD></TR>\n";
            os << "      <TR><TD>x4: " << fmtX4(v->position()) << "</TD></TR>\n";
            os << "      <TR><TD>nIn: " << v->particles_in().size() << " nOut: " << v->particles_out().size()
               << "</TD></TR>\n";
          }
        }
      }

      // --- SIM enrichment
      if (r.kind == TruthGraph::NodeKind::SimTrack && haveSim) {
        os << "      <TR><TD>p4: " << fmtP4(simt->momentum()) << "</TD></TR>\n";

        const int32_t gn = g.nodeSimTrackToGen(i);
        if (gn >= 0)
          os << "      <TR><TD>GenParticle nodeId: " << gn << "</TD></TR>\n";

        const int32_t vn = g.nodeSimTrackToVtx(i);
        if (vn >= 0)
          os << "      <TR><TD>SimVertex nodeId: " << vn << "</TD></TR>\n";

        if (crossedBoundary) {
          os << "      <TR><TD><FONT COLOR=\"red\">crossedBoundary: true"
             << " idAtBoundary=" << simt->getIDAtBoundary() << "</FONT></TD></TR>\n";
          os << "      <TR><TD><FONT COLOR=\"red\">x4@boundary: " << fmtX4(simt->getPositionAtBoundary())
             << "</FONT></TD></TR>\n";
          os << "      <TR><TD><FONT COLOR=\"red\">p4@boundary: " << fmtP4(simt->getMomentumAtBoundary())
             << "</FONT></TD></TR>\n";
        }
      }

      os << "    </TABLE>\n";
      os << "  >];\n";
    }

    // edges
    for (uint32_t src = 0; src < n; ++src) {
      const uint32_t b = g.edgeBegin(src);
      const uint32_t e = g.edgeEnd(src);

      unsigned kept = 0;
      for (uint32_t pos = b; pos < e; ++pos) {
        const uint32_t dst = g.edges[pos];
        if (dst >= n)
          continue;
        os << "  n" << src << " -> n" << dst << edgeAttrs(g.edgeKind[pos]) << ";\n";
        if (++kept >= maxEdgesPerNode_)
          break;
      }
    }

    os << "}\n";
    os.close();
  }

private:
  edm::EDGetTokenT<TruthGraph> token_;
  std::string dotFile_;
  unsigned maxNodes_;
  unsigned maxEdgesPerNode_;

  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken_;

  edm::EDGetTokenT<edm::HepMCProduct> hepmc2Token_;
  edm::EDGetTokenT<edm::HepMC3Product> hepmc3Token_;
};

DEFINE_FWK_MODULE(TruthGraphDumper);
