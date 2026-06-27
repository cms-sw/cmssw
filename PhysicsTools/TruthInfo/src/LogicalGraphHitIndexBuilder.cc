// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace truth {

  LogicalGraphHitIndexBuilder::LogicalGraphHitIndexBuilder(uint32_t nParticles)
      : nParticles_(nParticles), children_(nParticles) {
    for (auto& channel : directHits_)
      channel.resize(nParticles);
  }

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

  void LogicalGraphHitIndexBuilder::addHit(
      HitChannel channel, uint32_t trackId, uint32_t detId, float energy, uint32_t recHitIndex) {
    if (energy <= 0.f)
      return;

    auto it = trackIdToParticle_.find(trackId);
    if (it == trackIdToParticle_.end())
      return;

    const std::size_t ch = static_cast<std::size_t>(channel);
    appendHit(directHits_[ch][it->second], detId, recHitIndex, energy);
    channelTouched_[ch] = true;
  }

  void LogicalGraphHitIndexBuilder::appendHit(HitList& hits, uint32_t detId, uint32_t recHitIndex, float energy) {
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
    std::vector<LogicalGraphHitIndex::Channel> channels(kNumHitChannels);

    for (std::size_t ch = 0; ch < kNumHitChannels; ++ch) {
      // Empty channels (not selected / detector absent) stay default-constructed,
      // skipping the per-particle subgraph aggregation and CSR build entirely.
      if (!channelTouched_[ch])
        continue;

      auto& direct = directHits_[ch];

      // Coalesce the per-particle direct-hit lists once, so the subgraph
      // aggregation and the CSR build both operate on sorted, de-duplicated spans.
      for (auto& hits : direct)
        coalesce(hits);

      std::vector<HitList> subgraph(nParticles_);
      std::vector<uint8_t> state(nParticles_, 0);
      for (uint32_t particleId = 0; particleId < nParticles_; ++particleId)
        fillSubgraphHits(particleId, direct, subgraph, state);

      auto& out = channels[ch];
      buildHitCSR(direct, out.directOffsets, out.directHits);
      buildHitCSR(subgraph, out.subgraphOffsets, out.subgraphHits);
    }

    return LogicalGraphHitIndex(nParticles_, std::move(channels));
  }

}  // namespace truth
