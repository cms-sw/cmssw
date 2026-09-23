// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_SubgraphHitView_h
#define PhysicsTools_TruthInfo_SubgraphHitView_h

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  // A particle's subgraph hits, coalesced and detId-sorted, whichever layout the index
  // carries. Use this rather than LogicalGraphHitIndex::subgraphHits when the particle
  // can be any node of the graph: that accessor returns a single span, so under the
  // shared layout it is empty for a GEN-only particle, which spans several slot ranges.
  //
  // The materialised layout persists exactly this form, so it is handed back untouched.
  // The shared layout stores each hit once, in tree order and repeating a detId per
  // contributing descendant, so it is coalesced here and kept for the rest of the event.
  // A particle whose subgraph is a single one-slot range needs no work either way: the
  // builder already sorted and summed its direct hits.
  //
  // Hold one per event and per module. It caches, so it is not thread safe and must not
  // be a member of a global or stream module shared across concurrent events.
  class SubgraphHitView {
  public:
    using Hit = LogicalGraphHitIndex::Hit;

    explicit SubgraphHitView(LogicalGraphHitIndex const& hitIndex) : hitIndex_(&hitIndex) {}

    [[nodiscard]] std::span<const Hit> subgraphHits(HitChannel channel, uint32_t particleId);

    // Pass-throughs, so a caller that needs both can hold the view alone rather than the
    // view and the index side by side, which is how the wrong accessor gets used again.
    [[nodiscard]] std::span<const Hit> directHits(HitChannel channel, uint32_t particleId) const {
      return hitIndex_->directHits(channel, particleId);
    }
    [[nodiscard]] bool hasChannel(HitChannel channel) const { return hitIndex_->hasChannel(channel); }
    [[nodiscard]] uint32_t nParticles() const { return hitIndex_->nParticles(); }
    [[nodiscard]] LogicalGraphHitIndex const& index() const { return *hitIndex_; }

  private:
    [[nodiscard]] static uint64_t key(HitChannel channel, uint32_t particleId) {
      return (static_cast<uint64_t>(channel) << 32) | particleId;
    }

    LogicalGraphHitIndex const* hitIndex_;

    // One vector per coalesced particle, not offsets into a shared store: a store that
    // grows would reallocate and dangle every span already handed out. An unordered_map
    // node keeps its vector put, so a span stays valid for the life of the view.
    std::unordered_map<uint64_t, std::vector<Hit>> cached_;
  };

}  // namespace truth

#endif
