// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h

#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  class LogicalGraphHitIndexBuilder {
  public:
    // sharedSubgraphStore selects the shared layout described in LogicalGraphHitIndex:
    // each hit is stored once, in an order that makes every subtree a contiguous range,
    // instead of being copied into each ancestor's aggregate. This is what the producer
    // writes by default. False builds the materialised layout.
    explicit LogicalGraphHitIndexBuilder(uint32_t nParticles, bool sharedSubgraphStore = true);

    // trackId is event-local (each mixing sub-event reuses 1,2,3,...); it MUST be
    // namespaced by the packed EncodedEventId or signal and pileup collide.
    void setSimTrackForParticle(uint32_t particleId, uint64_t eventId, uint32_t trackId);
    void addParticleChild(uint32_t parentParticleId, uint32_t childParticleId);

    // Add a hit on `trackId`'s SimTrack to `channel`. recHitIndex defaults to "no
    // recHit" for channels without a DetId->RecHit link (tracker, muon); calo/MTD
    // pass the mapped global recHit index.
    void addHit(HitChannel channel,
                uint64_t eventId,
                uint32_t trackId,
                uint32_t detId,
                float energy,
                uint32_t recHitIndex = LogicalGraphHitIndex::Hit::kInvalidRecHitIndex);

    // (EncodedEventId, trackId) -> global map key. The packed EncodedEventId fits in
    // 32 bits (reco::EncodedEventId::rawId is uint32), so shift it into the high word.
    static uint64_t simKey(uint64_t eventId, uint32_t trackId) {
      return (eventId << 32) | static_cast<uint64_t>(trackId);
    }

    [[nodiscard]] LogicalGraphHitIndex finish();

    // Whether finish() actually wrote the shared layout. False when it was not asked
    // for, and also when it was asked for but the hit-carrying particles did not form
    // a forest, in which case finish() falls back to the materialised layout.
    [[nodiscard]] bool usedSharedStore() const { return usedSharedStore_; }

  private:
    using Hit = LogicalGraphHitIndex::Hit;
    using SlotRange = LogicalGraphHitIndex::SlotRange;

    // Per-particle hits are accumulated as a flat, append-only list and coalesced
    // (summed per detId, sorted) lazily. This keeps the hot insertion path a
    // single push_back and avoids a per-particle hash table (one per particle for
    // each channel), which dominated CPU and memory at high hit multiplicity.
    using HitList = std::vector<Hit>;

    static void appendHit(HitList& hits, uint32_t detId, uint32_t recHitIndex, float energy);

    // Sort by detId and merge entries that share a detId: energies are summed and
    // the recHitIndex is the unique valid index for that detId, if any (a detId
    // maps to a single recHit, so all valid entries agree). Entries that coalesce
    // to non-positive energy are dropped. Idempotent on already-coalesced lists.
    // Summation runs in detId order, so coalesced energies are deterministic and
    // independent of hit insertion order (unlike a hash-map accumulation, whose
    // sum order was bucket-dependent); cell energies can therefore differ from a
    // hash-based build at the float-reassociation level (~1e-7 relative).
    static void coalesce(HitList& hits);

    // Collect the particle and every distinct descendant (cycle-safe) into
    // `order`. `visited`/`touched`/`stack` are reusable scratch: `touched` lists
    // the ids set in `visited` so they can be cleared in O(subgraph size) between
    // calls. Each descendant appears exactly once, so a particle reachable through
    // several paths (a re-convergent DAG) is not double-counted when its direct
    // hits are later summed into the subgraph aggregate.
    void collectSubgraphParticles(uint32_t particleId,
                                  std::vector<uint8_t>& visited,
                                  std::vector<uint32_t>& touched,
                                  std::vector<uint32_t>& stack,
                                  std::vector<uint32_t>& order) const;

    // Concatenate the (already coalesced) per-particle lists into CSR storage.
    static void buildHitCSR(std::vector<HitList> const& lists,
                            std::vector<uint32_t>& offsets,
                            std::vector<Hit>& storage);

    // Order the particles so that every subtree occupies consecutive slots, which is
    // what lets a subgraph be a range of the single hit store. The tree is the SIM
    // parentage alone: only particles with a SimTrack carry hits, and each of those has
    // at most one parent that also has a SimTrack, whereas the GEN half above them is a
    // DAG whose vertices have several incoming particles. Fills the DFS slot of every
    // particle and the number of particles in its subtree.
    // Per particle, whether its descendant closure contains anything with a SimTrack.
    [[nodiscard]] std::vector<uint8_t> closureReachesSimTrack() const;

    // False when the hit-carrying particles do not form a forest the tree can carry,
    // either because one has two hit-carrying parents or because one has a GEN-only
    // child with hit-carrying descendants of its own; the outputs are then meaningless.
    [[nodiscard]] bool buildDfsOrder(std::vector<uint32_t>& slotToParticle,
                                     std::vector<uint32_t>& dfsPos,
                                     std::vector<uint32_t>& subtreeCount) const;

    // Slot ranges covering each particle's subgraph: one run for a hit-carrying
    // particle, the merged union of the runs below it for a GEN-only particle.
    void buildSubgraphRanges(std::vector<uint32_t> const& dfsPos,
                             std::vector<uint32_t> const& subtreeCount,
                             std::vector<uint32_t>& rangeOffsets,
                             std::vector<SlotRange>& ranges) const;

    [[nodiscard]] LogicalGraphHitIndex finishShared();
    [[nodiscard]] LogicalGraphHitIndex finishMaterialised();

    static constexpr uint32_t kNoParent = std::numeric_limits<uint32_t>::max();

    uint32_t nParticles_ = 0;
    bool sharedSubgraphStore_ = false;
    bool usedSharedStore_ = false;

    std::unordered_map<uint64_t, uint32_t> trackIdToParticle_;
    std::vector<std::vector<uint32_t>> children_;

    // Whether a particle has a SimTrack, so it can carry hits and take part in the
    // SIM parentage tree.
    std::vector<uint8_t> hasSimTrack_;

    // [channel index][particle] -> direct hit list. Subgraph hits are aggregated
    // in finish().
    std::array<std::vector<HitList>, kNumHitChannels> directHits_;

    // A channel that never received a hit (not selected, or its detector absent)
    // is left empty by finish() without the per-particle subgraph aggregation.
    std::array<bool, kNumHitChannels> channelTouched_{};
  };

}  // namespace truth

#endif
