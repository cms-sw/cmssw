// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// DQM performance plots for the truth::Branch graph as a replacement for the
// TrackingParticle in track->truth association - the tracker counterpart of
// BranchHGCalValidator. A TrackingParticle has no hits of its own, so the
// comparison is mediated by the reco track: for each reco track it (a) matches the
// track to a branch through the tracker simhit index (shared DetIds) and (b)
// matches it to a TrackingParticle through the standard ClusterTPAssociation, then
// maps that TP back to its logical particle via the SimTrack trackId. The branch
// reproduces the TP-based assignment when both point at the same logical particle.
// The booked numerator/denominator (TP-matched tracks vs Branch-and-TP-agree) are
// turned into a "reproduction efficiency vs eta/pt" by the harvester
// (DQMGenericClient); the shared-hit completeness is booked directly. This is the
// DQM form of BranchTrackerReplacementValidator and sits alongside the standard
// tracking validation so the two truth descriptions can be compared.

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/trackHitsToClusterRefs.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

class BranchTrackingValidator : public DQMEDAnalyzer {
public:
  explicit BranchTrackingValidator(edm::ParameterSet const&);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct Plots {
    // Numerator/denominator for the harvester-computed reproduction efficiency.
    MonitorElement* denomEta = nullptr;
    MonitorElement* denomPt = nullptr;
    MonitorElement* effNumEta = nullptr;
    MonitorElement* effNumPt = nullptr;
    // Quality distributions for the best branch match. (For tracking the best-match
    // Branch performance is exactly these, and the self-match rate is the efficiency
    // above: effnum = best Branch is the TrackingParticle's natural Branch.)
    MonitorElement* completenessHits = nullptr;
    MonitorElement* sharedHits = nullptr;
    MonitorElement* completenessVsEta = nullptr;
    MonitorElement* completenessVsPt = nullptr;
    // Merge/split: distinct Branches sharing >=10% of the track's hits.
    MonitorElement* nSharingBranches = nullptr;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
  const edm::EDGetTokenT<ClusterTPAssociation> clusterTPToken_;

  const std::string folder_;
  const double minPt_;
  const double maxEta_;

  Plots plots_;
};

BranchTrackingValidator::BranchTrackingValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      trackToken_(consumes<edm::View<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      clusterTPToken_(consumes<ClusterTPAssociation>(cfg.getParameter<edm::InputTag>("clusterTPMap"))),
      folder_(cfg.getParameter<std::string>("folder")),
      minPt_(cfg.getParameter<double>("minPt")),
      maxEta_(cfg.getParameter<double>("maxEta")) {}

void BranchTrackingValidator::bookHistograms(DQMStore::IBooker& ib, edm::Run const&, edm::EventSetup const&) {
  ib.setCurrentFolder(folder_ + "/TrackingParticle");

  constexpr int kEtaBins = 40;
  constexpr double kEtaMax = 3.2;
  constexpr int kPtBins = 50;
  constexpr double kPtMax = 200.;

  plots_.denomEta = ib.book1D("denom_eta", "TP-matched tracks vs #eta;#eta;tracks", kEtaBins, -kEtaMax, kEtaMax);
  plots_.denomPt = ib.book1D("denom_pt", "TP-matched tracks vs p_{T};p_{T} [GeV];tracks", kPtBins, 0., kPtMax);
  plots_.effNumEta =
      ib.book1D("effnum_eta", "Branch-reproduced TP assignment vs #eta;#eta;tracks", kEtaBins, -kEtaMax, kEtaMax);
  plots_.effNumPt =
      ib.book1D("effnum_pt", "Branch-reproduced TP assignment vs p_{T};p_{T} [GeV];tracks", kPtBins, 0., kPtMax);

  plots_.completenessHits =
      ib.book1D("completeness_hits", "Branch shared-hit completeness;shared hits / track hits;tracks", 52, -0.01, 1.03);
  plots_.sharedHits = ib.book1D("shared_hits", "Branch shared tracker hits;shared hits;tracks", 40, 0., 40.);
  plots_.completenessVsEta = ib.bookProfile("completeness_vs_eta",
                                            "Branch shared-hit completeness vs #eta;#eta;completeness",
                                            kEtaBins,
                                            -kEtaMax,
                                            kEtaMax,
                                            0.,
                                            1.05);
  plots_.completenessVsPt = ib.bookProfile("completeness_vs_pt",
                                           "Branch shared-hit completeness vs p_{T};p_{T} [GeV];completeness",
                                           kPtBins,
                                           0.,
                                           kPtMax,
                                           0.,
                                           1.05);
  plots_.nSharingBranches = ib.book1D(
      "n_sharing_branches", "Distinct Branches sharing >=10% of the track hits;#Branches;tracks", 11, -0.5, 10.5);
}

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

  // tightest (smallest tracker subgraph) among the best-scoring matches, and its
  // shared-hit count. Returns {-1, 0} when there is no match.
  struct BestMatch {
    int particle = -1;
    uint32_t sharedHits = 0;
  };

  BestMatch tightestBest(std::vector<truth::BranchMatch> const& matches, truth::LogicalGraphHitIndex const& hitIndex) {
    if (matches.empty())
      return {};
    const float best = matches.front().score;
    BestMatch out{static_cast<int>(matches.front().rootParticleId),
                  static_cast<uint32_t>(matches.front().sharedEnergy)};
    std::size_t tightestSize = hitIndex.subgraphHits(truth::HitChannel::Tracker, matches.front().rootParticleId).size();
    for (auto const& m : matches) {
      if (m.score > best)
        break;
      const std::size_t size = hitIndex.subgraphHits(truth::HitChannel::Tracker, m.rootParticleId).size();
      if (size < tightestSize) {
        tightestSize = size;
        out.particle = static_cast<int>(m.rootParticleId);
        out.sharedHits = static_cast<uint32_t>(m.sharedEnergy);
      }
    }
    return out;
  }
}  // namespace

void BranchTrackingValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& tracks = event.get(trackToken_);
  auto const& clusterTP = event.get(clusterTPToken_);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);
  truth::BranchHitAssociator assoc(
      hitIndex, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Tracker);

  for (auto const& track : tracks) {
    const double eta = track.eta();
    const double pt = track.pt();
    if (pt < minPt_ || std::abs(eta) > maxEta_)
      continue;

    // Branch side: reco-track rechit DetIds -> best (tightest) tracker branch.
    std::vector<truth::RecoHit> trackHits;
    uint32_t nTrackHits = 0;
    for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
      const TrackingRecHit* hit = &(**it);
      if (hit->isValid()) {
        trackHits.push_back(truth::RecoHit{hit->geographicalId().rawId(), 1.f, 1.f});
        ++nTrackHits;
      }
    }
    BestMatch branch;
    std::vector<truth::BranchMatch> matches;
    if (!trackHits.empty()) {
      matches = assoc.bestBranches(std::span<const truth::RecoHit>(trackHits));
      branch = tightestBest(matches, hitIndex);
    }

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

    // The TrackingParticle assignment is the reference the Branch should reproduce.
    if (expectedParticle < 0)
      continue;
    plots_.denomEta->Fill(eta);
    plots_.denomPt->Fill(pt);

    if (branch.particle >= 0 && nTrackHits > 0) {
      const double completeness = static_cast<double>(branch.sharedHits) / nTrackHits;
      plots_.completenessHits->Fill(completeness);
      plots_.sharedHits->Fill(branch.sharedHits);
      plots_.completenessVsEta->Fill(eta, completeness);
      plots_.completenessVsPt->Fill(pt, completeness);
    }

    // Merge/split: distinct Branches sharing >=10% of the track's hits.
    if (nTrackHits > 0) {
      const double shareThreshold = 0.1 * static_cast<double>(nTrackHits);
      uint32_t nSharing = 0;
      for (auto const& m : matches)
        if (static_cast<double>(m.sharedEnergy) >= shareThreshold)
          ++nSharing;
      plots_.nSharingBranches->Fill(std::min<uint32_t>(nSharing, 10));
    }

    if (branch.particle == expectedParticle) {
      plots_.effNumEta->Fill(eta);
      plots_.effNumPt->Fill(pt);
    }
  }
}

void BranchTrackingValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("clusterTPMap", edm::InputTag("tpClusterProducer"));
  desc.add<std::string>("folder", "Tracking/BranchValidator");
  desc.add<double>("minPt", 0.9);
  desc.add<double>("maxEta", 3.0);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(BranchTrackingValidator);
