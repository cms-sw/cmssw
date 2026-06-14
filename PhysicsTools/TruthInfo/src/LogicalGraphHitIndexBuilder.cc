// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace truth {

  LogicalGraphHitIndexBuilder::LogicalGraphHitIndexBuilder(uint32_t nParticles)
      : nParticles_(nParticles),
        children_(nParticles),
        directHits_(nParticles),
        subgraphHits_(nParticles),
        trackerDirectHits_(nParticles),
        trackerSubgraphHits_(nParticles) {}

  void LogicalGraphHitIndexBuilder::setSimTrackForParticle(uint32_t particleId, uint32_t trackId) {
    if (particleId >= nParticles_)
      return;

    trackIdToParticle_[trackId] = particleId;
  }

  void LogicalGraphHitIndexBuilder::addParticleChild(uint32_t parentParticleId, uint32_t childParticleId) {
    if (parentParticleId >= nParticles_ || childParticleId >= nParticles_)
      return;

    children_[parentParticleId].push_back(childParticleId);
  }

  void LogicalGraphHitIndexBuilder::addHitForTrack(uint32_t trackId,
                                                   uint32_t detId,
                                                   uint32_t recHitIndex,
                                                   float energy) {
    if (energy <= 0.f)
      return;

    auto it = trackIdToParticle_.find(trackId);
    if (it == trackIdToParticle_.end())
      return;

    addHit(directHits_[it->second], detId, recHitIndex, energy);
  }

  void LogicalGraphHitIndexBuilder::addTrackerHitForTrack(uint32_t trackId, uint32_t detId, float energy) {
    if (energy <= 0.f)
      return;

    auto it = trackIdToParticle_.find(trackId);
    if (it == trackIdToParticle_.end())
      return;

    addHit(trackerDirectHits_[it->second], detId, Hit::invalidRecHitIndex, energy);
  }

  void LogicalGraphHitIndexBuilder::addHit(HitList& hits, uint32_t detId, uint32_t recHitIndex, float energy) {
    hits.push_back(Hit{detId, recHitIndex, energy});
  }

  void LogicalGraphHitIndexBuilder::coalesce(HitList& hits) {
    if (hits.empty())
      return;

    // Sort by detId, then recHitIndex. invalidRecHitIndex == UINT32_MAX sorts
    // last, so the first entry of each detId run already carries the valid index
    // when one exists.
    std::sort(hits.begin(), hits.end(), [](Hit const& a, Hit const& b) {
      if (a.detId != b.detId)
        return a.detId < b.detId;
      return a.recHitIndex < b.recHitIndex;
    });

    // In-place merge of consecutive entries that share a detId.
    std::size_t w = 0;
    for (std::size_t r = 0; r < hits.size(); ++r) {
      if (w > 0 && hits[w - 1].detId == hits[r].detId) {
        hits[w - 1].energy += hits[r].energy;
        if (hits[w - 1].recHitIndex == Hit::invalidRecHitIndex && hits[r].recHitIndex != Hit::invalidRecHitIndex)
          hits[w - 1].recHitIndex = hits[r].recHitIndex;
      } else {
        hits[w++] = hits[r];
      }
    }
    hits.resize(w);

    hits.erase(std::remove_if(hits.begin(), hits.end(), [](Hit const& h) { return h.energy <= 0.f; }), hits.end());
  }

  void LogicalGraphHitIndexBuilder::fillSubgraphHits(uint32_t particleId,
                                                     std::vector<HitList> const& direct,
                                                     std::vector<HitList>& subgraph,
                                                     std::vector<uint8_t>& state) {
    if (particleId >= nParticles_)
      return;

    // 1 = on the current DFS stack (cycle guard), 2 = already aggregated.
    if (state[particleId] != 0)
      return;

    state[particleId] = 1;

    auto& out = subgraph[particleId];
    out.insert(out.end(), direct[particleId].begin(), direct[particleId].end());

    for (uint32_t childId : children_[particleId]) {
      if (childId >= nParticles_)
        continue;

      fillSubgraphHits(childId, direct, subgraph, state);
      out.insert(out.end(), subgraph[childId].begin(), subgraph[childId].end());
    }

    coalesce(out);
    state[particleId] = 2;
  }

  void LogicalGraphHitIndexBuilder::buildHitCSR(std::vector<HitList> const& lists,
                                                std::vector<uint32_t>& offsets,
                                                std::vector<Hit>& storage) {
    offsets.clear();
    storage.clear();
    offsets.reserve(lists.size() + 1);
    offsets.push_back(0);

    std::size_t total = 0;
    for (auto const& list : lists)
      total += list.size();
    storage.reserve(total);

    // Lists are already coalesced (sorted by detId, deduplicated), so the CSR is
    // a plain concatenation: each particle's span stays contiguous and ordered.
    for (auto const& list : lists) {
      storage.insert(storage.end(), list.begin(), list.end());
      offsets.push_back(static_cast<uint32_t>(storage.size()));
    }
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finish() {
    // Coalesce the per-particle direct-hit lists once, so the subgraph
    // aggregation and the CSR build both operate on sorted, de-duplicated spans.
    for (auto& hits : directHits_)
      coalesce(hits);
    for (auto& hits : trackerDirectHits_)
      coalesce(hits);

    std::vector<uint8_t> caloState(nParticles_, 0);
    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      fillSubgraphHits(particleId, directHits_, subgraphHits_, caloState);
    }

    std::vector<uint8_t> trackerState(nParticles_, 0);
    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      fillSubgraphHits(particleId, trackerDirectHits_, trackerSubgraphHits_, trackerState);
    }

    std::vector<uint32_t> directOffsets;
    std::vector<Hit> directHitStorage;
    buildHitCSR(directHits_, directOffsets, directHitStorage);

    std::vector<uint32_t> subgraphOffsets;
    std::vector<Hit> subgraphHitStorage;
    buildHitCSR(subgraphHits_, subgraphOffsets, subgraphHitStorage);

    std::vector<uint32_t> trackerDirectOffsets;
    std::vector<Hit> trackerDirectHitStorage;
    buildHitCSR(trackerDirectHits_, trackerDirectOffsets, trackerDirectHitStorage);

    std::vector<uint32_t> trackerSubgraphOffsets;
    std::vector<Hit> trackerSubgraphHitStorage;
    buildHitCSR(trackerSubgraphHits_, trackerSubgraphOffsets, trackerSubgraphHitStorage);

    return LogicalGraphHitIndex(nParticles_,
                                std::move(directOffsets),
                                std::move(directHitStorage),
                                std::move(subgraphOffsets),
                                std::move(subgraphHitStorage),
                                std::move(trackerDirectOffsets),
                                std::move(trackerDirectHitStorage),
                                std::move(trackerSubgraphOffsets),
                                std::move(trackerSubgraphHitStorage));
  }

}  // namespace truth
