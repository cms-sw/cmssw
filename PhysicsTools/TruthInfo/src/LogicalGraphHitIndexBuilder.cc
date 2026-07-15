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

  void LogicalGraphHitIndexBuilder::setSimTrackForParticle(uint32_t particleId, uint64_t eventId, uint32_t trackId) {
    if (particleId >= nParticles_)
      return;

    trackIdToParticle_[simKey(eventId, trackId)] = particleId;
  }

  void LogicalGraphHitIndexBuilder::addParticleChild(uint32_t parentParticleId, uint32_t childParticleId) {
    if (parentParticleId >= nParticles_ || childParticleId >= nParticles_)
      return;

    children_[parentParticleId].push_back(childParticleId);
  }

  void LogicalGraphHitIndexBuilder::addHit(
      HitChannel channel, uint64_t eventId, uint32_t trackId, uint32_t detId, float energy, uint32_t recHitIndex) {
    if (energy <= 0.f)
      return;

    auto it = trackIdToParticle_.find(simKey(eventId, trackId));
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

    // Sort by detId, then recHitIndex. kInvalidRecHitIndex == UINT32_MAX sorts
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
        if (hits[w - 1].recHitIndex == Hit::kInvalidRecHitIndex && hits[r].recHitIndex != Hit::kInvalidRecHitIndex)
          hits[w - 1].recHitIndex = hits[r].recHitIndex;
      } else {
        hits[w++] = hits[r];
      }
    }
    hits.resize(w);

    hits.erase(std::remove_if(hits.begin(), hits.end(), [](Hit const& h) { return h.energy <= 0.f; }), hits.end());
  }

  void LogicalGraphHitIndexBuilder::collectSubgraphParticles(uint32_t particleId,
                                                             std::vector<uint8_t>& visited,
                                                             std::vector<uint32_t>& touched,
                                                             std::vector<uint32_t>& stack,
                                                             std::vector<uint32_t>& order) const {
    order.clear();
    if (particleId >= nParticles_)
      return;

    // Iterative DFS over the distinct descendants, cycle-safe via `visited`. A
    // descendant reachable through more than one path (a re-convergent DAG, e.g. a
    // particle whose production vertex has several incoming particles that share a
    // common ancestor) is enqueued and summed only once; merging the already
    // aggregated child subgraphs instead would add such a descendant's per-cell
    // energy once per path (coalesce() sums equal detIds), inflating the subgraph.
    stack.clear();
    stack.push_back(particleId);
    visited[particleId] = 1;
    touched.push_back(particleId);

    while (!stack.empty()) {
      const uint32_t current = stack.back();
      stack.pop_back();
      order.push_back(current);

      for (uint32_t childId : children_[current]) {
        if (childId >= nParticles_ || visited[childId])
          continue;
        visited[childId] = 1;
        touched.push_back(childId);
        stack.push_back(childId);
      }
    }
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
      std::vector<uint8_t> visited(nParticles_, 0);
      std::vector<uint32_t> touched;
      std::vector<uint32_t> stack;
      std::vector<uint32_t> order;
      for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
        collectSubgraphParticles(particleId, visited, touched, stack, order);

        auto& out = subgraph[particleId];
        for (const uint32_t descendant : order)
          out.insert(out.end(), direct[descendant].begin(), direct[descendant].end());
        coalesce(out);

        // Reset only the entries we set, keeping the per-particle cost proportional
        // to the subgraph size rather than nParticles_.
        for (const uint32_t id : touched)
          visited[id] = 0;
        touched.clear();
      }

      auto& out = channels[ch];
      buildHitCSR(direct, out.directOffsets, out.directHits);
      buildHitCSR(subgraph, out.subgraphOffsets, out.subgraphHits);
    }

    return LogicalGraphHitIndex(nParticles_, std::move(channels));
  }

}  // namespace truth
