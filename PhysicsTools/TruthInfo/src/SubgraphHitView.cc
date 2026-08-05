// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/SubgraphHitView.h"

#include <algorithm>

namespace truth {

  std::span<const SubgraphHitView::Hit> SubgraphHitView::subgraphHits(HitChannel channel, uint32_t particleId) {
    if (!hitIndex_->sharedSubgraphStore())
      return hitIndex_->subgraphHits(channel, particleId);

    const auto ranges = hitIndex_->subgraphRanges(particleId);
    if (ranges.empty())
      return {};

    // A single one-slot range is one particle's own direct hits, which the builder
    // already sorted and summed per detId.
    if (ranges.size() == 1 && ranges[0].slotCount == 1)
      return hitIndex_->rangeHits(channel, ranges[0]);

    const uint64_t cacheKey = key(channel, particleId);
    const auto found = cached_.find(cacheKey);
    if (found != cached_.end())
      return found->second;

    std::vector<Hit> hits;
    hitIndex_->appendSubgraphHits(channel, particleId, hits);

    // The same rule LogicalGraphHitIndexBuilder::coalesce applies: sort by detId, sum
    // the energies that share one, and keep the valid recHit index, which sorts first
    // because the invalid sentinel is UINT32_MAX.
    std::sort(hits.begin(), hits.end(), [](Hit const& a, Hit const& b) {
      if (a.detId != b.detId)
        return a.detId < b.detId;
      return a.recHitIndex < b.recHitIndex;
    });
    std::size_t w = 0;
    for (std::size_t r = 0; r < hits.size(); ++r) {
      if (w > 0 && hits[w - 1].detId == hits[r].detId) {
        hits[w - 1].energy += hits[r].energy;
        if (hits[w - 1].recHitIndex == Hit::kInvalidRecHitIndex)
          hits[w - 1].recHitIndex = hits[r].recHitIndex;
      } else {
        hits[w++] = hits[r];
      }
    }
    hits.resize(w);

    return cached_.emplace(cacheKey, std::move(hits)).first->second;
  }

}  // namespace truth
