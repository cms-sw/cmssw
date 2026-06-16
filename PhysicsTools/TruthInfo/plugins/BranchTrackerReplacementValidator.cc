// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Validates that a truth::Branch can stand in for a TrackingParticle for
// track<->truth association. For each reco track it (a) matches the track to a
// Branch via shared tracker hits (our PSimHit-based tracker hit index, DetId
// keyed) and (b) matches it to a TrackingParticle via the existing
// ClusterTPAssociation (cluster->TP). If the Branch and the TP correspond to the
// same truth particle, the Branch reproduces the TP-based association.

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/trackHitsToClusterRefs.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

class BranchTrackerReplacementValidator : public edm::one::EDAnalyzer<> {
public:
  explicit BranchTrackerReplacementValidator(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPToken_;

  uint64_t nTracks_ = 0;
  uint64_t nBothMatched_ = 0;
  uint64_t nAgree_ = 0;
  uint64_t nBranchOnly_ = 0;
  uint64_t nTPOnly_ = 0;
};

BranchTrackerReplacementValidator::BranchTrackerReplacementValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      trackToken_(consumes<edm::View<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      clusterTPToken_(consumes<ClusterTPAssociation>(cfg.getParameter<edm::InputTag>("clusterTPMap"))) {}

namespace {
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

  // tightest (smallest tracker subgraph) among the best-scoring matches.
  int tightestBest(std::vector<truth::BranchMatch> const& matches, truth::LogicalGraphHitIndex const& hitIndex) {
    if (matches.empty())
      return -1;
    const float best = matches.front().score;
    int tightest = static_cast<int>(matches.front().rootParticleId);
    std::size_t tightestSize = hitIndex.subgraphHits(truth::HitChannel::Tracker, matches.front().rootParticleId).size();
    for (auto const& m : matches) {
      if (m.score > best)
        break;
      const std::size_t size = hitIndex.subgraphHits(truth::HitChannel::Tracker, m.rootParticleId).size();
      if (size < tightestSize) {
        tightestSize = size;
        tightest = static_cast<int>(m.rootParticleId);
      }
    }
    return tightest;
  }
}  // namespace

void BranchTrackerReplacementValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& tracks = event.get(trackToken_);
  auto const& clusterTP = event.get(clusterTPToken_);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);
  truth::BranchHitAssociator assoc(
      hitIndex, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Tracker);

  for (auto const& track : tracks) {
    ++nTracks_;

    // Branch side: reco-track rechit DetIds -> best tracker branch.
    std::vector<truth::RecoHit> trackHits;
    for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
      const TrackingRecHit* hit = &(**it);
      if (hit->isValid())
        trackHits.push_back(truth::RecoHit{hit->geographicalId().rawId(), 1.f, 1.f});
    }
    int branchParticle = -1;
    if (!trackHits.empty())
      branchParticle = tightestBest(assoc.bestBranches(std::span<const truth::RecoHit>(trackHits)), hitIndex);

    // TP side: shared clusters via ClusterTPAssociation -> dominant TP -> particle.
    auto clusters = track_associator::hitsToClusterRefs(track.recHitsBegin(), track.recHitsEnd());
    std::unordered_map<uint32_t, int> tpClusters;
    std::unordered_map<uint32_t, uint32_t> tpTrackId;
    for (auto const& omni : clusters) {
      auto range = clusterTP.equal_range(omni);
      for (auto i = range.first; i != range.second; ++i) {
        const auto& tpRef = i->second;
        const uint32_t key = tpRef.key();
        ++tpClusters[key];
        if (!tpTrackId.count(key) && !tpRef->g4Tracks().empty())
          tpTrackId[key] = tpRef->g4Tracks().front().trackId();
      }
    }
    int expectedParticle = -1;
    int bestShared = 0;
    for (auto const& [key, count] : tpClusters) {
      if (count > bestShared) {
        bestShared = count;
        auto tit = tidToParticle.find(tpTrackId[key]);
        expectedParticle = tit != tidToParticle.end() ? static_cast<int>(tit->second) : -1;
      }
    }

    if (branchParticle >= 0 && expectedParticle >= 0) {
      ++nBothMatched_;
      if (branchParticle == expectedParticle)
        ++nAgree_;
    } else if (branchParticle >= 0) {
      ++nBranchOnly_;
    } else if (expectedParticle >= 0) {
      ++nTPOnly_;
    }
  }
}

void BranchTrackerReplacementValidator::endJob() {
  const double agree = nBothMatched_ > 0 ? static_cast<double>(nAgree_) / nBothMatched_ : 0.0;
  edm::LogPrint("BranchTrackerReplacementValidator")
      << "=== Branch vs TrackingParticle (track->truth) replacement ===\n"
      << "tracks=" << nTracks_ << " bothMatched=" << nBothMatched_ << " branchOnly=" << nBranchOnly_
      << " tpOnly=" << nTPOnly_ << " | Branch-TP agreement=" << agree;
}

DEFINE_FWK_MODULE(BranchTrackerReplacementValidator);
