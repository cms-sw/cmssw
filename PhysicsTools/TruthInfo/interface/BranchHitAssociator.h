// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_interface_BranchHitAssociator_h
#define PhysicsTools_TruthInfo_interface_BranchHitAssociator_h

#include <cstdint>
#include <ranges>
#include <span>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

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
  // energy as flat, sorted arrays (binary-searched, no per-event hashing).
  // bestBranches() then answers any reco object via a merge-join of the object's
  // (sorted) hits with each candidate's sorted subgraph-hit span.
  class BranchHitAssociator {
  public:
    enum class Metric { SharedEnergy, SharedHits };

    explicit BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                 std::vector<uint32_t> candidateRoots = {},
                                 Metric metric = Metric::SharedEnergy,
                                 HitChannel channel = HitChannel::HGCalCalo);

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

    // Candidate roots whose subgraph touches a cell, by binary search; empty span
    // if the cell is untouched.
    [[nodiscard]] std::span<const uint32_t> rootsForCell(uint32_t detId) const;
    // Total sim energy on a cell (denominator for branch fractions), 0 if none.
    [[nodiscard]] float cellTotalEnergy(uint32_t detId) const;

    LogicalGraphHitIndex const* hitIndex_;
    Metric metric_;
    HitChannel channel_;
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
  };

}  // namespace truth

#endif
