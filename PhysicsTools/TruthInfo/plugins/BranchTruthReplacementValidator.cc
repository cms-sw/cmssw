// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Validates that a truth::Branch can stand in for the legacy calo truth objects
// (CaloParticle, SimCluster): for each legacy object it maps the object to its
// logical particle (via the SimTrack trackId), compares the Branch's subgraph
// calo hits to the object's hits_and_fractions (completeness/purity), and checks
// that the generic BranchHitAssociator picks that same Branch as the best match.
// A high completeness + correct best-match rate means the Branch reproduces the
// legacy object and can replace it.

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

class BranchTruthReplacementValidator : public edm::one::EDAnalyzer<> {
public:
  explicit BranchTruthReplacementValidator(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;

private:
  struct Stats {
    uint64_t n = 0;
    uint64_t unmapped = 0;
    uint64_t bestMatchCorrect = 0;
    double sumCompletenessHits = 0.;
    double sumPurityHits = 0.;
    double sumCompletenessEnergy = 0.;
    void print(const char* name) const;
  };

  template <class Collection>
  void validate(Collection const& objects,
                truth::Graph const& graph,
                TruthGraph const& raw,
                truth::LogicalGraphHitIndex const& hitIndex,
                truth::BranchHitAssociator const& assoc,
                std::unordered_map<uint32_t, uint32_t> const& tidToParticle,
                Stats& stats);

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;

  Stats caloParticleStats_;
  Stats simClusterStats_;
};

BranchTruthReplacementValidator::BranchTruthReplacementValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(cfg.getParameter<edm::InputTag>("caloParticles"))),
      simClusterToken_(consumes<std::vector<SimCluster>>(cfg.getParameter<edm::InputTag>("simClusters"))) {}

namespace {
  // logical-particle id <- SimTrack trackId, via the raw-graph node back-reference.
  std::unordered_map<uint32_t, uint32_t> buildTrackIdToParticle(truth::Graph const& graph, TruthGraph const& raw) {
    std::unordered_map<uint32_t, uint32_t> out;
    out.reserve(graph.nParticles());
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int32_t simNode = graph.particles[i].simNode;
      if (simNode < 0 || static_cast<uint32_t>(simNode) >= raw.nNodes())
        continue;
      auto const& nr = raw.nodeRef(static_cast<uint32_t>(simNode));
      if (nr.kind == TruthGraph::NodeKind::SimTrack)
        out[static_cast<uint32_t>(nr.key)] = i;
    }
    return out;
  }
}  // namespace

template <class Collection>
void BranchTruthReplacementValidator::validate(Collection const& objects,
                                               truth::Graph const& graph,
                                               TruthGraph const& raw,
                                               truth::LogicalGraphHitIndex const& hitIndex,
                                               truth::BranchHitAssociator const& assoc,
                                               std::unordered_map<uint32_t, uint32_t> const& tidToParticle,
                                               Stats& stats) {
  for (auto const& obj : objects) {
    if (obj.g4Tracks().empty())
      continue;
    ++stats.n;

    const uint32_t trackId = obj.g4Tracks().front().trackId();
    auto it = tidToParticle.find(trackId);
    if (it == tidToParticle.end()) {
      ++stats.unmapped;
      continue;
    }
    const uint32_t particleId = it->second;

    // Branch subgraph calo hits for the mapped logical particle.
    std::unordered_set<uint32_t> branchDetIds;
    for (auto const& hit : hitIndex.subgraphHits(particleId))
      branchDetIds.insert(hit.detId);

    // Legacy object hits, and the reco-like hit list for the matcher.
    auto hitsAndFractions = obj.hits_and_fractions();
    std::vector<truth::RecoHit> recoHits;
    recoHits.reserve(hitsAndFractions.size());
    uint32_t shared = 0;
    double totalFraction = 0.;
    double sharedFraction = 0.;
    for (auto const& [detId, fraction] : hitsAndFractions) {
      recoHits.push_back(truth::RecoHit{detId, 1.f, fraction});
      totalFraction += fraction;
      if (branchDetIds.count(detId)) {
        ++shared;
        sharedFraction += fraction;
      }
    }
    if (hitsAndFractions.empty())
      continue;

    stats.sumCompletenessHits += static_cast<double>(shared) / hitsAndFractions.size();
    stats.sumPurityHits += branchDetIds.empty() ? 0. : static_cast<double>(shared) / branchDetIds.size();
    stats.sumCompletenessEnergy += totalFraction > 0. ? sharedFraction / totalFraction : 0.;

    // Does the associator identify this particle's branch? Many ancestor
    // branches fully contain the object (same best score), so among the
    // best-scoring matches pick the tightest (smallest subgraph) -- that is the
    // particle itself, not a broader ancestor.
    auto matches = assoc.bestBranches(std::span<const truth::RecoHit>(recoHits));
    if (!matches.empty()) {
      const float bestScore = matches.front().score;
      uint32_t tightest = matches.front().rootParticleId;
      std::size_t tightestSize = hitIndex.subgraphHits(tightest).size();
      for (auto const& m : matches) {
        if (m.score > bestScore)
          break;
        const std::size_t size = hitIndex.subgraphHits(m.rootParticleId).size();
        if (size < tightestSize) {
          tightestSize = size;
          tightest = m.rootParticleId;
        }
      }
      if (tightest == particleId)
        ++stats.bestMatchCorrect;
    }
  }
}

void BranchTruthReplacementValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);

  // SharedHits metric: best branch = the one sharing the most calo cells.
  truth::BranchHitAssociator assoc(hitIndex, {}, truth::BranchHitAssociator::Metric::SharedHits);

  validate(event.get(caloParticleToken_), graph, raw, hitIndex, assoc, tidToParticle, caloParticleStats_);
  validate(event.get(simClusterToken_), graph, raw, hitIndex, assoc, tidToParticle, simClusterStats_);
}

void BranchTruthReplacementValidator::Stats::print(const char* name) const {
  if (n == 0) {
    edm::LogPrint("BranchTruthReplacementValidator") << name << ": no objects";
    return;
  }
  const uint64_t mapped = n - unmapped;
  const double inv = mapped > 0 ? 1.0 / static_cast<double>(mapped) : 0.0;
  edm::LogPrint("BranchTruthReplacementValidator")
      << name << ": N=" << n << " mapped=" << mapped << " unmapped=" << unmapped
      << " | mean hit-completeness=" << sumCompletenessHits * inv
      << " mean energy-completeness=" << sumCompletenessEnergy * inv << " mean purity=" << sumPurityHits * inv
      << " | best-branch-correct=" << (mapped > 0 ? static_cast<double>(bestMatchCorrect) / mapped : 0.0);
}

void BranchTruthReplacementValidator::endJob() {
  edm::LogPrint("BranchTruthReplacementValidator") << "=== Branch vs legacy calo truth replacement ===";
  caloParticleStats_.print("CaloParticle");
  simClusterStats_.print("SimCluster");
}

DEFINE_FWK_MODULE(BranchTruthReplacementValidator);
