#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
#include <utility>

namespace truth {

  LogicalGraphHitIndexBuilder::LogicalGraphHitIndexBuilder(uint32_t nParticles)
      : nParticles_(nParticles), children_(nParticles), directHits_(nParticles), subgraphHits_(nParticles) {}

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

  void LogicalGraphHitIndexBuilder::addHit(HitMap& hits, uint32_t detId, uint32_t recHitIndex, float energy) {
    auto& entry = hits[detId];
    entry.energy += energy;

    if (entry.recHitIndex == Hit::invalidRecHitIndex && recHitIndex != Hit::invalidRecHitIndex) {
      entry.recHitIndex = recHitIndex;
    }
  }

  void LogicalGraphHitIndexBuilder::addHit(HitMap& hits, Hit const& hit) {
    addHit(hits, hit.detId, hit.recHitIndex, hit.energy);
  }

  std::vector<LogicalGraphHitIndexBuilder::Hit> LogicalGraphHitIndexBuilder::sortedHits(HitMap const& hits) {
    std::vector<Hit> out;
    out.reserve(hits.size());

    for (auto const& [detId, acc] : hits) {
      if (acc.energy <= 0.f)
        continue;

      Hit hit;
      hit.detId = detId;
      hit.recHitIndex = acc.recHitIndex;
      hit.energy = acc.energy;
      out.push_back(hit);
    }

    std::sort(out.begin(), out.end(), [](Hit const& a, Hit const& b) {
      if (a.detId != b.detId)
        return a.detId < b.detId;
      return a.recHitIndex < b.recHitIndex;
    });

    return out;
  }

  void LogicalGraphHitIndexBuilder::fillSubgraphHits(uint32_t particleId, std::vector<uint8_t>& state) {
    if (particleId >= nParticles_)
      return;

    if (state[particleId] == 2)
      return;

    if (state[particleId] == 1)
      return;

    state[particleId] = 1;

    auto& out = subgraphHits_[particleId];

    for (auto const& [detId, acc] : directHits_[particleId]) {
      addHit(out, detId, acc.recHitIndex, acc.energy);
    }

    for (uint32_t childId : children_[particleId]) {
      if (childId >= nParticles_)
        continue;

      fillSubgraphHits(childId, state);

      for (auto const& [detId, acc] : subgraphHits_[childId]) {
        addHit(out, detId, acc.recHitIndex, acc.energy);
      }
    }

    state[particleId] = 2;
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finish() {
    std::vector<uint8_t> state(nParticles_, 0);
    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      fillSubgraphHits(particleId, state);
    }

    std::vector<uint32_t> directOffsets;
    std::vector<Hit> directHitStorage;
    directOffsets.reserve(nParticles_ + 1);
    directOffsets.push_back(0);

    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      auto hits = sortedHits(directHits_[particleId]);
      directHitStorage.insert(directHitStorage.end(), hits.begin(), hits.end());
      directOffsets.push_back(static_cast<uint32_t>(directHitStorage.size()));
    }

    std::vector<uint32_t> subgraphOffsets;
    std::vector<Hit> subgraphHitStorage;
    subgraphOffsets.reserve(nParticles_ + 1);
    subgraphOffsets.push_back(0);

    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      auto hits = sortedHits(subgraphHits_[particleId]);
      subgraphHitStorage.insert(subgraphHitStorage.end(), hits.begin(), hits.end());
      subgraphOffsets.push_back(static_cast<uint32_t>(subgraphHitStorage.size()));
    }

    return LogicalGraphHitIndex(nParticles_,
                                std::move(directOffsets),
                                std::move(directHitStorage),
                                std::move(subgraphOffsets),
                                std::move(subgraphHitStorage));
  }

}  // namespace truth
