#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
#include <stdexcept>

namespace truth {

  void LogicalGraphHitIndexBuilder::reset(uint32_t nParticles) {
    nParticles_ = nParticles;

    trackIdToParticle_.clear();
    trackIdToParticle_.reserve(nParticles * 2);

    children_.assign(nParticles, {});
    directHits_.assign(nParticles, {});
    subgraphHits_.assign(nParticles, {});

    totalSimHitEnergyByDetId_.clear();
  }

  void LogicalGraphHitIndexBuilder::setSimTrackForParticle(uint32_t particleId, uint32_t trackId) {
    if (particleId >= nParticles_)
      throw std::out_of_range("LogicalGraphHitIndexBuilder::setSimTrackForParticle: particle index out of range");

    trackIdToParticle_.emplace(trackId, particleId);
  }

  void LogicalGraphHitIndexBuilder::addParticleChild(uint32_t parentParticleId, uint32_t childParticleId) {
    if (parentParticleId >= nParticles_ || childParticleId >= nParticles_)
      throw std::out_of_range("LogicalGraphHitIndexBuilder::addParticleChild: particle index out of range");

    children_[parentParticleId].push_back(childParticleId);
  }

  void LogicalGraphHitIndexBuilder::addHitForTrack(uint32_t trackId, uint32_t detId, float energy) {
    if (energy == 0.f)
      return;

    auto const it = trackIdToParticle_.find(trackId);
    if (it == trackIdToParticle_.end())
      return;

    addHitForParticle(it->second, detId, energy);
  }

  void LogicalGraphHitIndexBuilder::addHitForParticle(uint32_t particleId, uint32_t detId, float energy) {
    if (particleId >= nParticles_)
      throw std::out_of_range("LogicalGraphHitIndexBuilder::addHitForParticle: particle index out of range");

    if (energy == 0.f)
      return;

    directHits_[particleId].push_back({detId, energy});
    totalSimHitEnergyByDetId_.push_back({detId, energy});
  }

  void LogicalGraphHitIndexBuilder::sortAndReduce(std::vector<Hit>& hits) {
    std::sort(hits.begin(), hits.end(), [](Hit const& a, Hit const& b) { return a.detId < b.detId; });

    std::size_t out = 0;
    for (auto const& hit : hits) {
      if (hit.energy == 0.f)
        continue;

      if (out != 0 && hits[out - 1].detId == hit.detId) {
        hits[out - 1].energy += hit.energy;
      } else {
        hits[out++] = hit;
      }
    }

    hits.resize(out);
  }

  void LogicalGraphHitIndexBuilder::sortAndReduceDirectHits() {
    for (auto& hits : directHits_) {
      sortAndReduce(hits);
    }

    sortAndReduce(totalSimHitEnergyByDetId_);

    for (auto& daughters : children_) {
      std::sort(daughters.begin(), daughters.end());
      daughters.erase(std::unique(daughters.begin(), daughters.end()), daughters.end());
    }
  }

  std::vector<LogicalGraphHitIndexBuilder::Hit> LogicalGraphHitIndexBuilder::mergeSortedHitLists(
      std::span<const Hit> a, std::span<const Hit> b) {
    std::vector<Hit> out;
    out.reserve(a.size() + b.size());

    auto ia = a.begin();
    auto ib = b.begin();

    while (ia != a.end() && ib != b.end()) {
      if (ia->detId < ib->detId) {
        out.push_back(*ia++);
      } else if (ib->detId < ia->detId) {
        out.push_back(*ib++);
      } else {
        out.push_back({ia->detId, ia->energy + ib->energy});
        ++ia;
        ++ib;
      }
    }

    out.insert(out.end(), ia, a.end());
    out.insert(out.end(), ib, b.end());

    return out;
  }

  void LogicalGraphHitIndexBuilder::mergeInto(std::vector<Hit>& dst, std::span<const Hit> src) {
    if (src.empty())
      return;

    if (dst.empty()) {
      dst.assign(src.begin(), src.end());
      return;
    }

    dst = mergeSortedHitLists(dst, src);
  }

  void LogicalGraphHitIndexBuilder::collectSubgraphParticles(uint32_t rootParticleId,
                                                             std::vector<uint8_t>& visited) const {
    std::vector<uint32_t> stack;
    stack.push_back(rootParticleId);

    while (!stack.empty()) {
      const uint32_t particleId = stack.back();
      stack.pop_back();

      if (particleId >= nParticles_ || visited[particleId])
        continue;

      visited[particleId] = 1;

      for (uint32_t child : children_[particleId]) {
        stack.push_back(child);
      }
    }
  }

  void LogicalGraphHitIndexBuilder::computeSubgraphHits() {
    subgraphHits_.assign(nParticles_, {});

    std::vector<uint8_t> visited;

    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      visited.assign(nParticles_, 0);
      collectSubgraphParticles(particleId, visited);

      auto& out = subgraphHits_[particleId];

      for (uint32_t other = 0; other < nParticles_; ++other) {
        if (!visited[other])
          continue;

        mergeInto(out, directHits_[other]);
      }
    }
  }

  void LogicalGraphHitIndexBuilder::buildCsr(std::vector<std::vector<Hit>> const& perParticleHits,
                                             std::vector<uint32_t>& offsets,
                                             std::vector<Hit>& storage) const {
    offsets.clear();
    storage.clear();

    offsets.reserve(nParticles_ + 1);
    offsets.push_back(0);

    std::size_t totalSize = 0;
    for (auto const& hits : perParticleHits) {
      totalSize += hits.size();
    }
    storage.reserve(totalSize);

    for (auto const& hits : perParticleHits) {
      storage.insert(storage.end(), hits.begin(), hits.end());
      offsets.push_back(static_cast<uint32_t>(storage.size()));
    }
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finish() {
    sortAndReduceDirectHits();
    computeSubgraphHits();

    std::vector<uint32_t> directOffsets;
    std::vector<Hit> directStorage;
    buildCsr(directHits_, directOffsets, directStorage);

    std::vector<uint32_t> subgraphOffsets;
    std::vector<Hit> subgraphStorage;
    buildCsr(subgraphHits_, subgraphOffsets, subgraphStorage);

    LogicalGraphHitIndex out;
    out.setData(nParticles_,
                std::move(directOffsets),
                std::move(directStorage),
                std::move(subgraphOffsets),
                std::move(subgraphStorage),
                std::move(totalSimHitEnergyByDetId_));

    return out;
  }

  std::vector<LogicalGraphHitIndexBuilder::Hit> LogicalGraphHitIndexBuilder::collectMergedDirectHits(
      std::span<const uint32_t> particles) const {
    std::vector<Hit> out;

    for (uint32_t particleId : particles) {
      if (particleId >= nParticles_)
        continue;

      mergeInto(out, directHits_[particleId]);
    }

    return out;
  }

  std::vector<LogicalGraphHitIndexBuilder::Hit> LogicalGraphHitIndexBuilder::collectMergedSubgraphHits(
      std::span<const uint32_t> particles) const {
    std::vector<uint8_t> visited(nParticles_, 0);

    for (uint32_t particleId : particles) {
      if (particleId < nParticles_)
        collectSubgraphParticles(particleId, visited);
    }

    std::vector<Hit> out;

    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      if (!visited[particleId])
        continue;

      mergeInto(out, directHits_[particleId]);
    }

    return out;
  }

}  // namespace truth
