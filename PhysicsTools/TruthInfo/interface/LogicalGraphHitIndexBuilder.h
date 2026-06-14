// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  class LogicalGraphHitIndexBuilder {
  public:
    explicit LogicalGraphHitIndexBuilder(uint32_t nParticles);

    void setSimTrackForParticle(uint32_t particleId, uint32_t trackId);
    void addParticleChild(uint32_t parentParticleId, uint32_t childParticleId);

    void addHitForTrack(uint32_t trackId, uint32_t detId, uint32_t recHitIndex, float energy);

    // Tracker simhits: separate channel, no recHit association.
    void addTrackerHitForTrack(uint32_t trackId, uint32_t detId, float energy);

    [[nodiscard]] LogicalGraphHitIndex finish();

  private:
    using Hit = LogicalGraphHitIndex::Hit;

    // Per-particle hits are accumulated as a flat, append-only list and coalesced
    // (summed per detId, sorted) lazily. This keeps the hot insertion path a
    // single push_back and avoids a per-particle hash table (one per particle for
    // each of the four channels), which dominated CPU and memory at high hit
    // multiplicity.
    using HitList = std::vector<Hit>;

    static void addHit(HitList& hits, uint32_t detId, uint32_t recHitIndex, float energy);

    // Sort by detId and merge entries that share a detId: energies are summed and
    // the recHitIndex is the unique valid index for that detId, if any (a detId
    // maps to a single recHit, so all valid entries agree). Entries that coalesce
    // to non-positive energy are dropped. Idempotent on already-coalesced lists.
    // Summation runs in detId order, so coalesced energies are deterministic and
    // independent of hit insertion order (unlike a hash-map accumulation, whose
    // sum order was bucket-dependent); cell energies can therefore differ from a
    // hash-based build at the float-reassociation level (~1e-7 relative).
    static void coalesce(HitList& hits);

    // Aggregate the (already coalesced) direct hits of a particle and all its
    // descendants into subgraph[particleId]; the result is coalesced in place.
    void fillSubgraphHits(uint32_t particleId,
                          std::vector<HitList> const& direct,
                          std::vector<HitList>& subgraph,
                          std::vector<uint8_t>& state);

    // Concatenate the (already coalesced) per-particle lists into CSR storage.
    static void buildHitCSR(std::vector<HitList> const& lists,
                            std::vector<uint32_t>& offsets,
                            std::vector<Hit>& storage);

    uint32_t nParticles_ = 0;

    std::unordered_map<uint32_t, uint32_t> trackIdToParticle_;
    std::vector<std::vector<uint32_t>> children_;

    std::vector<HitList> directHits_;
    std::vector<HitList> subgraphHits_;

    std::vector<HitList> trackerDirectHits_;
    std::vector<HitList> trackerSubgraphHits_;
  };

}  // namespace truth

#endif
