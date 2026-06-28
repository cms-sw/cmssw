// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Diagnostic analyzer that audits the raw TruthGraph and the logical truth::Graph
// for "strange" topologies and reports their structural provenance:
//   * vertices with many outgoing particles (hadronization, showers, system vtx);
//   * particles with more than one parent particle (hard scatter, multi-mother);
//   * particles produced at more than one vertex (structural anomaly);
//   * cycles (the graphs must be DAGs);
//   * disconnected components (orphans).
// Everything is aggregated per job and printed in endJob, with a few worst-case
// examples carrying pdgId/status so the cause can be identified.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

namespace {

  // Running distribution of a non-negative integer quantity.
  struct DegreeStats {
    std::map<uint32_t, uint64_t> dist;
    uint32_t maxValue = 0;
    uint64_t sum = 0;
    uint64_t count = 0;
    void add(uint32_t v) {
      ++dist[v];
      maxValue = std::max(maxValue, v);
      sum += v;
      ++count;
    }
    [[nodiscard]] double mean() const { return count ? static_cast<double>(sum) / count : 0.0; }
    [[nodiscard]] uint64_t atLeast(uint32_t threshold) const {
      uint64_t n = 0;
      for (auto const& [v, c] : dist)
        if (v >= threshold)
          n += c;
      return n;
    }
    [[nodiscard]] std::string summary() const {
      std::ostringstream os;
      os << "count=" << count << " mean=" << mean() << " max=" << maxValue
         << " | n(=0)=" << (dist.count(0) ? dist.at(0) : 0) << " n(=1)=" << (dist.count(1) ? dist.at(1) : 0)
         << " n(2-9)=" << (atLeast(2) - atLeast(10)) << " n(10-49)=" << (atLeast(10) - atLeast(50))
         << " n(>=50)=" << atLeast(50);
      return os.str();
    }
  };

  // Union-find for weakly-connected components.
  struct UnionFind {
    std::vector<uint32_t> parent;
    explicit UnionFind(uint32_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    uint32_t find(uint32_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    }
    void unite(uint32_t a, uint32_t b) { parent[find(a)] = find(b); }
  };

  // Kahn topological sort over a directed graph given as out-adjacency + indegree.
  // Returns true if it is a DAG (all nodes processed).
  bool isDag(std::vector<std::vector<uint32_t>> const& outAdj, std::vector<uint32_t> indeg) {
    std::vector<uint32_t> stack;
    for (uint32_t i = 0; i < indeg.size(); ++i)
      if (indeg[i] == 0)
        stack.push_back(i);
    uint64_t processed = 0;
    while (!stack.empty()) {
      const uint32_t u = stack.back();
      stack.pop_back();
      ++processed;
      for (const uint32_t v : outAdj[u])
        if (--indeg[v] == 0)
          stack.push_back(v);
    }
    return processed == indeg.size();
  }

  void capPush(std::vector<std::string>& v, std::string s, std::size_t cap = 6) {
    if (v.size() < cap)
      v.push_back(std::move(s));
  }

  // Decode the packed EncodedEventId (memcpy reverse of TruthGraphProducer::packEventId).
  EncodedEventId decodeEid(uint64_t packed) {
    uint32_t raw = 0;
    std::memcpy(&raw, &packed, sizeof(raw));
    return EncodedEventId(raw);
  }

}  // namespace

class TruthGraphTopologyChecker : public edm::one::EDAnalyzer<> {
public:
  explicit TruthGraphTopologyChecker(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;

private:
  void analyzeRaw(TruthGraph const&);
  void analyzeLogical(truth::Graph const&);

  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::Graph> logicalToken_;
  // When true, throw at endJob if any history-fragmentation violation (orphan
  // components or cycles, raw or logical) was seen - used by the history-guard
  // unit test to fail if the simulation stops producing connected parentage.
  const bool failOnViolations_;

  uint64_t nEvents_ = 0;

  // ---- raw graph accumulators ----
  DegreeStats rawGenVtxOut_, rawSimVtxOut_, rawGenVtxInMothers_, rawSimVtxInMothers_;
  DegreeStats rawGenProdVtx_, rawSimProdVtx_, rawGenParents_, rawSimParents_;
  uint64_t rawMultiProdParticles_ = 0, rawMultiMotherVertices_ = 0, rawMultiParentParticles_ = 0;
  uint64_t rawCycles_ = 0, rawComponentsTotal_ = 0, rawOrphanComponents_ = 0;
  std::vector<std::string> rawBigVtxEx_, rawMultiParentEx_, rawMultiProdEx_, rawOrphanEx_;

  // ---- logical graph accumulators ----
  DegreeStats logVtxOut_, logVtxInMothers_, logProdVtx_, logParents_, logDecayVtx_;
  uint64_t logMultiProdParticles_ = 0, logMultiMotherVertices_ = 0, logMultiParentParticles_ = 0;
  uint64_t logMultiDecayParticles_ = 0, logArtificialVertices_ = 0;
  uint64_t logCycles_ = 0, logComponentsTotal_ = 0, logOrphanComponents_ = 0;
  uint64_t logSignalParticles_ = 0, logPileupParticles_ = 0;
  std::map<int, uint64_t> logBxHist_;  // bunchCrossing -> particle count
  std::vector<std::string> logBigVtxEx_, logMultiParentEx_, logMultiProdEx_, logOrphanEx_;
};

TruthGraphTopologyChecker::TruthGraphTopologyChecker(edm::ParameterSet const& cfg)
    : rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      logicalToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      failOnViolations_(cfg.getUntrackedParameter<bool>("failOnViolations", false)) {}

void TruthGraphTopologyChecker::analyze(edm::Event const& event, edm::EventSetup const&) {
  ++nEvents_;
  analyzeRaw(event.get(rawToken_));
  analyzeLogical(event.get(logicalToken_));
}

void TruthGraphTopologyChecker::analyzeRaw(TruthGraph const& g) {
  using NK = TruthGraph::NodeKind;
  using EK = TruthGraph::EdgeKind;
  const uint32_t n = g.nNodes();

  // Reverse pass: classify incoming edges of each node by the source's kind and
  // by edge kind (structural Gen/Sim vs cross-realm GenToSim).
  std::vector<uint32_t> inFromVertex(n, 0), inFromParticle(n, 0);
  std::vector<int32_t> firstProdVertex(n, -1);
  std::vector<std::vector<uint32_t>> structOut(n);  // structural out-adjacency for DAG/components
  UnionFind uf(n);

  for (uint32_t s = 0; s < n; ++s) {
    const auto kids = g.children(s);
    const auto kinds = g.childrenEdgeKinds(s);
    const NK sk = g.nodeRef(s).kind;
    for (std::size_t e = 0; e < kids.size(); ++e) {
      const uint32_t d = kids[e];
      const EK ek = static_cast<EK>(kinds[e]);
      uf.unite(s, d);  // weak connectivity uses every edge, incl. GenToSim
      if (ek == EK::GenToSim || ek == EK::SimToGen)
        continue;
      structOut[s].push_back(d);
      if (sk == NK::GenVertex || sk == NK::SimVertex) {
        ++inFromVertex[d];
        if (firstProdVertex[d] < 0)
          firstProdVertex[d] = static_cast<int32_t>(s);
      } else if (sk == NK::GenParticle || sk == NK::SimTrack) {
        ++inFromParticle[d];
      }
    }
  }

  auto nodeStr = [&](uint32_t id) {
    std::ostringstream os;
    os << "pdg=" << g.nodePdgId(id) << " st=" << g.nodeStatus(id);
    return os.str();
  };

  for (uint32_t v = 0; v < n; ++v) {
    const NK k = g.nodeRef(v).kind;
    if (k == NK::GenVertex || k == NK::SimVertex) {
      const uint32_t outdeg = static_cast<uint32_t>(structOut[v].size());
      const uint32_t mothers = inFromParticle[v];
      const bool isGen = (k == NK::GenVertex);
      (isGen ? rawGenVtxOut_ : rawSimVtxOut_).add(outdeg);
      (isGen ? rawGenVtxInMothers_ : rawSimVtxInMothers_).add(mothers);
      if (mothers > 1)
        ++rawMultiMotherVertices_;
      if (outdeg >= 20) {
        std::ostringstream os;
        os << (isGen ? "GenVtx" : "SimVtx") << " out=" << outdeg << " mothers=" << mothers << " [";
        // mothers' pdgs: scan for particle nodes whose structural out-edge hits v
        for (uint32_t s = 0, shown = 0; s < n && shown < 4; ++s) {
          for (const uint32_t d : structOut[s])
            if (d == v && (g.nodeRef(s).kind == NK::GenParticle || g.nodeRef(s).kind == NK::SimTrack)) {
              os << nodeStr(s) << "; ";
              ++shown;
              break;
            }
        }
        os << "] sampleOut: ";
        for (uint32_t j = 0; j < outdeg && j < 5; ++j)
          os << nodeStr(structOut[v][j]) << "; ";
        capPush(rawBigVtxEx_, os.str());
      }
    } else if (k == NK::GenParticle || k == NK::SimTrack) {
      const bool isGen = (k == NK::GenParticle);
      const uint32_t prod = inFromVertex[v];
      (isGen ? rawGenProdVtx_ : rawSimProdVtx_).add(prod);
      if (prod > 1) {
        ++rawMultiProdParticles_;
        std::ostringstream os;
        os << (isGen ? "GenPart " : "SimTrk ") << nodeStr(v) << " nProdVtx=" << prod;
        capPush(rawMultiProdEx_, os.str());
      }
      if (prod >= 1 && firstProdVertex[v] >= 0) {
        const uint32_t pv = static_cast<uint32_t>(firstProdVertex[v]);
        const uint32_t parents = inFromParticle[pv];
        (isGen ? rawGenParents_ : rawSimParents_).add(parents);
        if (parents > 1) {
          ++rawMultiParentParticles_;
          std::ostringstream os;
          os << (isGen ? "GenPart " : "SimTrk ") << nodeStr(v) << " parents=" << parents << " [";
          for (uint32_t s = 0, shown = 0; s < n && shown < 5; ++s)
            for (const uint32_t d : structOut[s])
              if (d == pv && (g.nodeRef(s).kind == NK::GenParticle || g.nodeRef(s).kind == NK::SimTrack)) {
                os << nodeStr(s) << "; ";
                ++shown;
                break;
              }
          os << "]";
          capPush(rawMultiParentEx_, os.str());
        }
      }
    }
  }

  // DAG check over structural edges.
  std::vector<uint32_t> indeg(n, 0);
  for (uint32_t s = 0; s < n; ++s)
    for (const uint32_t d : structOut[s])
      ++indeg[d];
  if (!isDag(structOut, indeg))
    ++rawCycles_;

  // Weakly-connected components.
  std::map<uint32_t, uint32_t> compSize;
  for (uint32_t i = 0; i < n; ++i)
    ++compSize[uf.find(i)];
  rawComponentsTotal_ += compSize.size();
  for (auto const& [root, sz] : compSize)
    if (sz < n / 2 && sz <= 50) {  // small fragment relative to the event = orphan
      ++rawOrphanComponents_;
      if (rawOrphanEx_.size() < 6) {
        std::ostringstream os;
        os << "size=" << sz << " e.g. " << nodeStr(root);
        capPush(rawOrphanEx_, os.str());
      }
    }
}

void TruthGraphTopologyChecker::analyzeLogical(truth::Graph const& g) {
  const uint32_t nP = g.nParticles();
  const uint32_t nV = g.nVertices();
  const uint32_t n = nP + nV;  // combined indexing: particle p -> p, vertex v -> nP+v

  UnionFind uf(n);
  std::vector<std::vector<uint32_t>> outAdj(n);
  std::vector<uint32_t> indeg(n, 0);

  auto partStr = [&](uint32_t p) {
    std::ostringstream os;
    os << "pdg=" << g.particles()[p].pdgId << " st=" << g.particles()[p].status;
    return os.str();
  };

  for (uint32_t v = 0; v < nV; ++v) {
    if (g.vertices()[v].isArtificial())
      ++logArtificialVertices_;
    const auto out = g.outgoingParticles(v);
    const auto in = g.incomingParticles(v);
    logVtxOut_.add(static_cast<uint32_t>(out.size()));
    logVtxInMothers_.add(static_cast<uint32_t>(in.size()));
    if (in.size() > 1)
      ++logMultiMotherVertices_;
    for (const uint32_t p : out) {
      outAdj[nP + v].push_back(p);
      ++indeg[p];
      uf.unite(nP + v, p);
    }
    for (const uint32_t p : in) {
      outAdj[p].push_back(nP + v);
      ++indeg[nP + v];
      uf.unite(p, nP + v);
    }
    if (out.size() >= 20) {
      std::ostringstream os;
      os << "vtx out=" << out.size() << " in=" << in.size() << " mothers:[";
      for (std::size_t j = 0; j < in.size() && j < 4; ++j)
        os << partStr(in[j]) << "; ";
      os << "] sampleOut:[";
      for (std::size_t j = 0; j < out.size() && j < 5; ++j)
        os << partStr(out[j]) << "; ";
      os << "]";
      capPush(logBigVtxEx_, os.str());
    }
  }

  for (uint32_t p = 0; p < nP; ++p) {
    const auto prod = g.productionVertices(p);
    const auto decay = g.decayVertices(p);
    logProdVtx_.add(static_cast<uint32_t>(prod.size()));
    logDecayVtx_.add(static_cast<uint32_t>(decay.size()));
    if (prod.size() > 1) {
      ++logMultiProdParticles_;
      std::ostringstream os;
      os << partStr(p) << " nProdVtx=" << prod.size();
      capPush(logMultiProdEx_, os.str());
    }
    if (decay.size() > 1)
      ++logMultiDecayParticles_;
    uint32_t parents = 0;
    for (const uint32_t v : prod)
      parents += static_cast<uint32_t>(g.incomingParticles(v).size());
    logParents_.add(parents);
    if (parents > 1) {
      ++logMultiParentParticles_;
      std::ostringstream os;
      os << partStr(p) << " parents=" << parents << " [";
      for (const uint32_t v : prod)
        for (const uint32_t m : g.incomingParticles(v))
          os << partStr(m) << "; ";
      os << "]";
      capPush(logMultiParentEx_, os.str());
    }

    // Pileup provenance: signal is (bx==0, event==0); everything else is pileup.
    const EncodedEventId eid = decodeEid(g.particles()[p].eventId);
    ++logBxHist_[eid.bunchCrossing()];
    if (eid.bunchCrossing() == 0 && eid.event() == 0)
      ++logSignalParticles_;
    else
      ++logPileupParticles_;
  }

  if (!isDag(outAdj, indeg))
    ++logCycles_;

  std::map<uint32_t, uint32_t> compSize;
  for (uint32_t i = 0; i < n; ++i)
    ++compSize[uf.find(i)];
  logComponentsTotal_ += compSize.size();
  for (auto const& [root, sz] : compSize)
    if (sz < n / 2 && sz <= 50) {
      ++logOrphanComponents_;
      if (logOrphanEx_.size() < 6) {
        std::ostringstream os;
        os << "size=" << sz << (root < nP ? " (particle root)" : " (vertex root)");
        capPush(logOrphanEx_, os.str());
      }
    }
}

void TruthGraphTopologyChecker::endJob() {
  auto dump = [](const char* tag, std::vector<std::string> const& ex) {
    if (ex.empty())
      return;
    std::ostringstream os;
    os << tag;
    for (auto const& s : ex)
      os << "\n    - " << s;
    edm::LogPrint("TruthGraphTopologyChecker") << os.str();
  };

  edm::LogPrint("TruthGraphTopologyChecker")
      << "================ TruthGraph topology audit (" << nEvents_ << " events) ================\n"
      << "[RAW] GenVtx out-degree:   " << rawGenVtxOut_.summary() << "\n"
      << "[RAW] SimVtx out-degree:   " << rawSimVtxOut_.summary() << "\n"
      << "[RAW] GenVtx mothers(in):  " << rawGenVtxInMothers_.summary() << "\n"
      << "[RAW] SimVtx mothers(in):  " << rawSimVtxInMothers_.summary() << "\n"
      << "[RAW] GenPart prod-vtx:    " << rawGenProdVtx_.summary() << "\n"
      << "[RAW] SimTrk  prod-vtx:    " << rawSimProdVtx_.summary() << "\n"
      << "[RAW] GenPart parent-cnt:  " << rawGenParents_.summary() << "\n"
      << "[RAW] SimTrk  parent-cnt:  " << rawSimParents_.summary() << "\n"
      << "[RAW] anomalies: multiProdParticles=" << rawMultiProdParticles_
      << " multiMotherVertices=" << rawMultiMotherVertices_ << " multiParentParticles=" << rawMultiParentParticles_
      << " cyclesEvents=" << rawCycles_ << " components(sum)=" << rawComponentsTotal_
      << " orphanFragments=" << rawOrphanComponents_;
  dump("[RAW] big-out-degree vertices:", rawBigVtxEx_);
  dump("[RAW] multi-parent particles:", rawMultiParentEx_);
  dump("[RAW] multi-production particles:", rawMultiProdEx_);
  dump("[RAW] orphan fragments:", rawOrphanEx_);

  edm::LogPrint("TruthGraphTopologyChecker")
      << "---------------- logical truth::Graph ----------------\n"
      << "[LOG] Vtx out-degree:      " << logVtxOut_.summary() << "\n"
      << "[LOG] Vtx mothers(in):     " << logVtxInMothers_.summary() << "\n"
      << "[LOG] Part prod-vtx:       " << logProdVtx_.summary() << "\n"
      << "[LOG] Part decay-vtx:      " << logDecayVtx_.summary() << "\n"
      << "[LOG] Part parent-cnt:     " << logParents_.summary() << "\n"
      << "[LOG] anomalies: multiProdParticles=" << logMultiProdParticles_
      << " multiDecayParticles=" << logMultiDecayParticles_ << " multiMotherVertices=" << logMultiMotherVertices_
      << " multiParentParticles=" << logMultiParentParticles_ << " artificialVertices=" << logArtificialVertices_
      << " cyclesEvents=" << logCycles_ << " components(sum)=" << logComponentsTotal_
      << " orphanFragments=" << logOrphanComponents_;
  dump("[LOG] big-out-degree vertices:", logBigVtxEx_);
  dump("[LOG] multi-parent particles:", logMultiParentEx_);
  dump("[LOG] multi-production particles:", logMultiProdEx_);
  dump("[LOG] orphan fragments:", logOrphanEx_);

  std::ostringstream bx;
  for (auto const& [b, c] : logBxHist_)
    bx << " bx" << b << "=" << c;
  edm::LogPrint("TruthGraphTopologyChecker")
      << "[LOG] pileup provenance: signalParticles(bx=0,ev=0)=" << logSignalParticles_
      << " pileupParticles=" << logPileupParticles_ << " | per-bunchCrossing:" << bx.str();

  // History guard: a disconnected (orphan) fragment or a parentage cycle means the
  // SimTrack/SimVertex history no longer forms one tree reaching the generator -
  // exactly the regression a simulation change that drops the per-track parentage
  // (e.g. a GPU port) would cause. Fail hard when asked to (the history-guard test).
  if (failOnViolations_) {
    const uint64_t violations = rawOrphanComponents_ + logOrphanComponents_ + rawCycles_ + logCycles_;
    if (violations != 0)
      throw cms::Exception("TruthGraphHistoryBroken")
          << "Truth-graph history is fragmented over " << nEvents_
          << " events: raw orphanFragments=" << rawOrphanComponents_ << " cyclesEvents=" << rawCycles_
          << ", logical orphanFragments=" << logOrphanComponents_ << " cyclesEvents=" << logCycles_
          << ". The SimTrack/SimVertex parentage is no longer fully connected to the generator.";
  }
}

DEFINE_FWK_MODULE(TruthGraphTopologyChecker);
