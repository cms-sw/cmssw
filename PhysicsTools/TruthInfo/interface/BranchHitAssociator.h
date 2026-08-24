// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_BranchHitAssociator_h
#define PhysicsTools_TruthInfo_interface_BranchHitAssociator_h

#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  // The hit format the graph matches against. Any reco object can be matched by
  // exposing its hits as a range of RecoHit.
  struct RecoHit {
    uint32_t detId = 0;
    // The cell (rec)hit energy, for callers that need a per-object weight. The
    // SharedEnergy metric does not read it: it takes the cell energy from the truth
    // hit index, because a calorimetric adapter has no per-cell reco energy to give.
    float energy = 0.f;
    float fraction = 1.f;  // fraction of the cell assigned to this reco object
  };

  // Customization point: a reco object R is matchable if it exposes its hits via
  // a member R::truthHits() returning a range of RecoHit. A user wanting to match
  // their own reco object to the truth graph only needs to add this one method.
  template <class R>
  concept HasTruthHits = requires(const R& r) {
    { r.truthHits() } -> std::ranges::range;
  };

  struct BranchMatch {
    static constexpr uint32_t kInvalidRoot = std::numeric_limits<uint32_t>::max();
    uint32_t rootParticleId = 0;
    float sharedEnergy = 0.f;  // (SharedHits metric: number of shared cells)
    // Reco-normalized score: how much of the reco object the branch fails to
    // cover (denominator = reco self-energy / reco hit count). Use for the
    // reco->branch direction. Lower is better.
    float score = 0.f;
    // Branch-normalized score: how much of the branch the reco object fails to
    // cover (denominator = branch subgraph self-energy / branch hit count). Use
    // for the branch->reco direction. Lower is better.
    float reverseScore = 0.f;
    // Sim-normalized shared quantity: sharedEnergy over the branch's own energy IN
    // THE DETECTORS the caller asked the denominator to cover (its cell count for
    // SharedHits). This is the axis HGCalValidator gates efficiency on, and it is NOT
    // one minus reverseScore: the score is a squared, energy-weighted quantity, this
    // one is linear.
    float sharedEnergyFraction = 0.f;
  };

  // Lower score is better in every association map this package sorts. One shared
  // comparator so the [0]-is-best contract cannot drift between producers.
  inline constexpr auto byAscendingScore = [](const auto& a, const auto& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };

  // Associates reco objects to truth branches (subtrees) by shared detector hits.
  // Built once per event over a set of candidate branch roots (default: every
  // particle); caches the inverted detId -> roots index and per-cell total sim
  // energy as flat, sorted arrays (binary-searched, no per-event hashing).
  // bestBranches() then answers any reco object via a merge-join of the object's
  // (sorted) hits with each candidate's sorted subgraph-hit span.
  class BranchHitAssociator {
  public:
    // SharedEnergy reproduces the TICL trackster-to-simTrackster arithmetic
    // (SimCalorimetry/HGCalAssociatorProducers/plugins/
    // AllTracksterToSimTracksterAssociatorsByHitsProducer.cc:341-364 for reco->sim and
    // :428-453 for sim->reco): per cell the score is the squared uncovered energy over
    // the squared self energy, and the shared energy is the minimum of the two sides.
    // SharedHits counts cells and ignores energy, which is what the tracker needs.
    enum class Metric { SharedEnergy, SharedHits };

    // Detectors the sharedEnergyFraction denominator covers, as a bit per DetId::det()
    // value. One hit channel spans several detectors: HitChannel::Calo carries the
    // barrel ECAL and HCAL PCaloHits next to the HGCAL ones, and their sampling
    // fractions differ by orders of magnitude, so a branch that showered in the barrel
    // has a channel-wide energy no endcap reco object can ever reach a half of. The
    // caller passes the detectors its reco collection reconstructs and the fraction is
    // normalized to the branch energy there. kAllDetectors keeps the whole channel.
    static constexpr uint32_t kAllDetectors = 0xFFFFu;
    [[nodiscard]] static uint32_t detectorBit(uint32_t detId);

    // candidateRoots restricts the branch roots considered. By default an empty
    // list means "every particle" (the common unrestricted case). Pass
    // emptyRootsMeansAll = false to instead treat an empty list as "no candidates"
    // (match nothing) - needed when a caller asked for a restriction that happened
    // to select no particle in this event, which must not silently fall back to all.
    explicit BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                 std::vector<uint32_t> candidateRoots = {},
                                 Metric metric = Metric::SharedEnergy,
                                 HitChannel channel = HitChannel::Calo,
                                 bool emptyRootsMeansAll = true,
                                 uint32_t denominatorDetectors = kAllDetectors);

    // Best branches for a reco object's hits, sorted by score ascending. If
    // maxResults > 0, only the best maxResults are returned.
    [[nodiscard]] std::vector<BranchMatch> bestBranches(std::span<const RecoHit> recoHits,
                                                        std::size_t maxResults = 0) const;

    template <HasTruthHits R>
    [[nodiscard]] std::vector<BranchMatch> bestBranches(R const& reco, std::size_t maxResults = 0) const {
      std::vector<RecoHit> hits;
      for (auto const& h : reco.truthHits())
        hits.push_back(RecoHit{h.detId, h.energy, h.fraction});
      return bestBranches(std::span<const RecoHit>(hits), maxResults);
    }

    // Adaptive-level match. The candidates are every root that shares hits with the reco
    // object: the leaves and their ancestors, when the candidate set is the ancestor
    // closure. This returns the one candidate that minimises
    //     score + reverseWeight * reverseScore
    // As a branch climbs, score falls, because the branch covers more of the reco object.
    // At the same time reverseScore rises, because the branch spreads to energy the reco
    // object does not have. The minimum is the level that best matches the object.
    // Candidates whose reverseScore exceeds maxReverseScore (the branch-spread /
    // contamination ceiling) are rejected; if that empties the set, the ceiling is
    // ignored and the global minimum is returned. rootParticleId is
    // BranchMatch::kInvalidRoot if the reco object shares no hits with any root.
    [[nodiscard]] BranchMatch bestAdaptiveBranch(std::span<const RecoHit> recoHits,
                                                 float reverseWeight = 1.f,
                                                 float maxReverseScore = 1.f) const;

    // The same argmin over an already-computed bestBranches() list, so a caller
    // evaluating several working points on one object pays the merge-join once.
    [[nodiscard]] static BranchMatch bestAdaptiveBranch(std::span<const BranchMatch> matches,
                                                        float reverseWeight,
                                                        float maxReverseScore);

  private:
    // Fill the coalesced per-root hit store used by the shared layout. A no-op for a
    // materialised index, which already persists the coalesced spans.
    void buildRootHits();

    [[nodiscard]] std::span<const LogicalGraphHitIndex::Hit> rootHits(uint32_t rootId) const;

    // Candidate roots whose subgraph touches a cell, by binary search; empty span
    // if the cell is untouched.
    [[nodiscard]] std::span<const uint32_t> rootsForCell(uint32_t detId) const;
    // Total sim energy on a cell (denominator for branch fractions), 0 if none.
    [[nodiscard]] float cellTotalEnergy(uint32_t detId) const;

    LogicalGraphHitIndex const* hitIndex_;
    Metric metric_;
    HitChannel channel_;
    uint32_t denominatorDetectors_;
    std::vector<uint32_t> roots_;

    // Inverted index detId -> candidate roots, stored CSR-style: cellRootsKeys_
    // holds the distinct cell detIds (ascending); cellRootsOffsets_ indexes
    // cellRoots_, which holds the root ids (ascending within each cell).
    std::vector<uint32_t> cellRootsKeys_;
    std::vector<uint32_t> cellRootsOffsets_;
    std::vector<uint32_t> cellRoots_;

    // Per-cell total sim energy as parallel sorted arrays (cellEnergyKeys_ ascending).
    std::vector<uint32_t> cellEnergyKeys_;
    std::vector<float> cellEnergyValues_;

    // Per-root branch self-energy (sum of subgraph-hit energy^2 on channel_),
    // indexed by particle id; the denominator for the branch-normalized reverse
    // score. Computed once with the inverted index so bestBranches() needs no
    // full branch-hit scan.
    std::vector<double> rootSelfEnergySq_;
    // Per-root branch total energy (LINEAR sum of the same hits, restricted to
    // denominatorDetectors_), the denominator of sharedEnergyFraction.
    std::vector<double> rootEnergy_;

    // Shared layout only: the candidate roots' subgraph hits, coalesced here once at
    // construction because the persisted store keeps them in tree order with a detId
    // repeated per contributing descendant, while the merge-join below needs one
    // ascending entry per detId. Materialised indices keep using the persisted spans,
    // so these stay empty. CSR over roots_, in the order roots_ holds them.
    std::vector<uint32_t> rootHitOffsets_;
    std::vector<LogicalGraphHitIndex::Hit> rootHitStorage_;
    // particle id -> position in rootHitOffsets_, or kNoRoot when the particle is not a
    // candidate, or kPersistedSpan when the persisted span is already coalesced and no
    // private copy was made.
    std::vector<uint32_t> rootHitSlotOfRoot_;
    static constexpr uint32_t kNoRoot = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kPersistedSpan = kNoRoot - 1u;
  };

}  // namespace truth

#endif
