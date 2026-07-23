// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Builds TICL-style AssociationMaps between reco tracks and the truth::Branch
// graph, the tracker counterpart of TruthBranchCaloAssociationProducer. Unlike the
// calo truth objects (CaloParticle / SimCluster), a TrackingParticle carries no
// hits of its own - it is just a bundle of SimTracks - so it cannot be matched to a
// branch by shared cells directly. The hit-bearing probe is therefore the
// reco::Track: each track is matched to a branch through the tracker simhit index
// (PSimHit DetIds, the same channel the Branch tracker subgraph is built from),
// using the shared-hit-multiplicity metric (the tracker has no per-cell energy to
// share, unlike the calorimeter). The track<->TrackingParticle link that closes the
// Branch<->TrackingParticle comparison is provided by the standard ClusterTPAssociation
// (tpClusterProducer) and consumed downstream by BranchTrackingValidator; here we
// produce only the genuinely new, hit-based reco-track<->branch association. The
// branch side is restricted to the "interesting" particles via interestingPdgIds
// (empty = all). For each direction the matched branches/tracks are stored with
// their shared-hit count (in the sharedEnergy slot) and score (lower = better, i.e.
// more of the track's hits captured), sorted best first.

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  // Raw-index AssociationMap (object index -> [(branch id, sharedHits, score)]).
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
}  // namespace

class TruthBranchTrackingAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit TruthBranchTrackingAssociationProducer(edm::ParameterSet const&);
  void produce(edm::Event&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
  const std::vector<int> interestingPdgIds_;
};

TruthBranchTrackingAssociationProducer::TruthBranchTrackingAssociationProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      trackToken_(consumes<edm::View<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      interestingPdgIds_(cfg.getParameter<std::vector<int>>("interestingPdgIds")) {
  produces<BranchAssociationMap>("trackToBranch");
  produces<BranchAssociationMap>("branchToTrack");
}

void TruthBranchTrackingAssociationProducer::produce(edm::Event& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& tracks = event.get(trackToken_);

  // Candidate branch roots = the interesting particles. No configured restriction
  // means "all branches"; a configured restriction that matches nothing in this
  // event means "no branches" (not all), so the empty-roots fallback is disabled
  // whenever interestingPdgIds_ is set.
  const bool restrictRoots = !interestingPdgIds_.empty();
  std::vector<uint32_t> roots;
  if (restrictRoots) {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int pdgId = graph.particles()[i].pdgId;
      if (std::find(interestingPdgIds_.begin(), interestingPdgIds_.end(), pdgId) != interestingPdgIds_.end())
        roots.push_back(i);
    }
  }

  // SharedHits metric on the tracker channel: the tracker carries no per-cell
  // energy to share, so matches are ranked by the multiplicity of shared simhit
  // DetIds (score = fraction of the track's hits left uncaptured, lower = better).
  truth::BranchHitAssociator assoc(hitIndex,
                                   roots,
                                   truth::BranchHitAssociator::Metric::SharedHits,
                                   truth::HitChannel::Tracker,
                                   /*emptyRootsMeansAll=*/!restrictRoots);

  auto trackToBranch = std::make_unique<BranchAssociationMap>(static_cast<unsigned int>(tracks.size()));
  auto branchToTrack = std::make_unique<BranchAssociationMap>(graph.nParticles());

  std::vector<truth::RecoHit> recoHits;
  for (unsigned int trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
    auto const& track = tracks[trackIndex];

    // The track's hits as DetIds; the tracker channel has no per-cell energy, so
    // every hit weighs the same (energy = fraction = 1).
    recoHits.clear();
    for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
      const TrackingRecHit* hit = &(**it);
      if (hit->isValid())
        recoHits.push_back(truth::RecoHit{hit->geographicalId().rawId(), 1.f, 1.f});
    }

    if (recoHits.empty())
      continue;

    for (auto const& m : assoc.bestBranches(std::span<const truth::RecoHit>(recoHits))) {
      trackToBranch->insert(trackIndex, m.rootParticleId, m.sharedEnergy, m.score);
      // Branch-normalized score for the sim->reco direction (how much of the branch
      // this track captures), not the reco-normalized one.
      branchToTrack->insert(m.rootParticleId, trackIndex, m.sharedEnergy, m.reverseScore);
    }
  }

  // Sort each row by score in ascending order so the best-matched branch/track
  // is first. TICLAssociationMap::sort(true) sorts *descending* by score, but the
  // association score is lower-is-better, so we pass an explicit ascending
  // comparator (matching the standard HGCal associators).
  auto byAscendingScore = [](auto const& a, auto const& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };
  trackToBranch->sort(byAscendingScore);
  branchToTrack->sort(byAscendingScore);

  event.put(std::move(trackToBranch), "trackToBranch");
  event.put(std::move(branchToTrack), "branchToTrack");
}

void TruthBranchTrackingAssociationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<std::vector<int>>("interestingPdgIds", {})
      ->setComment("Restrict the branch side to these PDG ids (the interesting particles); empty = all branches.");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TruthBranchTrackingAssociationProducer);
