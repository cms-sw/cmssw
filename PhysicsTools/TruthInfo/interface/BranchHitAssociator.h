#ifndef PhysicsTools_TruthInfo_interface_BranchHitAssociator_h
#define PhysicsTools_TruthInfo_interface_BranchHitAssociator_h

#include <cstdint>
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  // The hit format the graph matches against. Any reco object can be matched by
  // exposing its hits as a range of RecoHit.
  struct RecoHit {
    uint32_t detId = 0;
    float energy = 0.f;    // the cell (rec)hit energy
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
    uint32_t rootParticleId = 0;
    float sharedEnergy = 0.f;  // (SharedHits metric: number of shared cells)
    float score = 0.f;         // lower is better
  };

  // Associates reco objects to truth branches (subtrees) by shared detector hits.
  // Built once per event over a set of candidate branch roots (default: every
  // particle); caches the inverted detId -> roots index and per-cell total sim
  // energy. bestBranches() then answers any reco object via a merge-join of the
  // object's (sorted) hits with each candidate's sorted subgraph-hit span.
  class BranchHitAssociator {
  public:
    enum class Metric { SharedEnergy, SharedHits };

    explicit BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                 std::vector<uint32_t> candidateRoots = {},
                                 Metric metric = Metric::SharedEnergy,
                                 bool useTracker = false);

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

  private:
    [[nodiscard]] std::span<const LogicalGraphHitIndex::Hit> rootHits(uint32_t rootId) const;

    LogicalGraphHitIndex const* hitIndex_;
    Metric metric_;
    bool useTracker_;
    std::vector<uint32_t> roots_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> detIdToRoots_;  // inverted index
    std::unordered_map<uint32_t, float> cellTotalEnergy_;               // per detId, summed sim energy
  };

}  // namespace truth

#endif
