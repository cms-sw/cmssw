// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
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

  void LogicalGraphHitIndexBuilder::fillSubgraphHits(uint32_t particleId,
                                                     std::vector<HitMap> const& direct,
                                                     std::vector<HitMap>& subgraph,
                                                     std::vector<uint8_t>& state) {
    if (particleId >= nParticles_)
      return;

    if (state[particleId] == 2)
      return;

    if (state[particleId] == 1)
      return;

    state[particleId] = 1;

    auto& out = subgraph[particleId];

    for (auto const& [detId, acc] : direct[particleId]) {
      addHit(out, detId, acc.recHitIndex, acc.energy);
    }

    for (uint32_t childId : children_[particleId]) {
      if (childId >= nParticles_)
        continue;

      fillSubgraphHits(childId, direct, subgraph, state);

      for (auto const& [detId, acc] : subgraph[childId]) {
        addHit(out, detId, acc.recHitIndex, acc.energy);
      }
    }

    state[particleId] = 2;
  }

  void LogicalGraphHitIndexBuilder::buildHitCSR(std::vector<HitMap> const& maps,
                                                std::vector<uint32_t>& offsets,
                                                std::vector<Hit>& storage) {
    offsets.clear();
    storage.clear();
    offsets.reserve(maps.size() + 1);
    offsets.push_back(0);

    for (auto const& map : maps) {
      auto hits = sortedHits(map);
      storage.insert(storage.end(), hits.begin(), hits.end());
      offsets.push_back(static_cast<uint32_t>(storage.size()));
    }
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finish() {
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
