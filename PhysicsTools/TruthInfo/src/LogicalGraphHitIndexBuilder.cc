// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace truth {

  LogicalGraphHitIndexBuilder::LogicalGraphHitIndexBuilder(uint32_t nParticles, bool sharedSubgraphStore)
      : nParticles_(nParticles),
        sharedSubgraphStore_(sharedSubgraphStore),
        children_(nParticles),
        hasSimTrack_(nParticles, 0) {
    for (auto& channel : directHits_)
      channel.resize(nParticles);
  }

  void LogicalGraphHitIndexBuilder::setSimTrackForParticle(uint32_t particleId, uint64_t eventId, uint32_t trackId) {
    if (particleId >= nParticles_)
      return;

    trackIdToParticle_[simKey(eventId, trackId)] = particleId;
    hasSimTrack_[particleId] = 1;
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

  std::vector<uint8_t> LogicalGraphHitIndexBuilder::closureReachesSimTrack() const {
    std::vector<uint8_t> reaches(nParticles_, 0);
    std::vector<uint8_t> state(nParticles_, 0);  // 0 = new, 1 = in progress, 2 = done
    std::vector<uint32_t> stack;

    for (uint32_t seed = 0; seed < nParticles_; ++seed) {
      if (state[seed] == 2)
        continue;
      stack.push_back(seed);

      while (!stack.empty()) {
        const uint32_t particleId = stack.back();

        if (state[particleId] == 2) {
          stack.pop_back();
          continue;
        }

        if (state[particleId] == 0) {
          state[particleId] = 1;
          for (const uint32_t child : children_[particleId]) {
            if (child < nParticles_ && state[child] == 0)
              stack.push_back(child);
          }
          continue;
        }

        // Second visit. A child still in progress is a cycle, and what lies beyond it is
        // unknown at this point, so it must count as REACHING: memoizing 0 here would be
        // wrong whenever the cycle has an exit to a SimTrack, and this function guards
        // the layout, so it over-approximates. The cost of a false positive is a
        // fallback to the materialised layout, which is always correct.
        uint8_t value = hasSimTrack_[particleId];
        for (const uint32_t child : children_[particleId]) {
          if (child >= nParticles_)
            continue;
          if (state[child] == 1 || (state[child] == 2 && reaches[child] != 0))
            value = 1;
        }
        reaches[particleId] = value;
        state[particleId] = 2;
        stack.pop_back();
      }
    }
    return reaches;
  }

  bool LogicalGraphHitIndexBuilder::buildDfsOrder(std::vector<uint32_t>& slotToParticle,
                                                  std::vector<uint32_t>& dfsPos,
                                                  std::vector<uint32_t>& subtreeCount) const {
    slotToParticle.clear();
    slotToParticle.reserve(nParticles_);
    dfsPos.assign(nParticles_, 0);
    subtreeCount.assign(nParticles_, 1);

    // A particle's SIM parent is the one parent that also has a SimTrack. Particles
    // without one are roots: the SIM primaries, and the GEN-only nodes, which carry no
    // hits and so become single-slot trees here.
    //
    // A subtree is only a contiguous run of slots when the hit-carrying particles form
    // a forest. A particle with two SimTrack parents would sit under one of them and be
    // missing from the other's run, so the layout cannot represent it and the caller
    // falls back to the materialised one.
    // Whether a particle's descendant closure contains anything that carries hits. Used
    // below to reject the one topology the tree cannot represent.
    const std::vector<uint8_t> reachesSim = closureReachesSimTrack();

    std::vector<uint32_t> simParent(nParticles_, kNoParent);
    std::vector<std::vector<uint32_t>> simChildren(nParticles_);
    bool isForest = true;
    for (uint32_t parent = 0; parent < nParticles_; ++parent) {
      if (hasSimTrack_[parent] == 0)
        continue;
      for (const uint32_t child : children_[parent]) {
        if (child >= nParticles_)
          continue;
        if (hasSimTrack_[child] == 0) {
          // A GEN-only child that still has hit-carrying descendants would put them
          // outside this parent's subtree run, since the tree is built from
          // hasSimTrack-to-hasSimTrack edges only, and the parent's subgraph would come
          // back short. No current sample does this, because a SIM-continued GEN
          // particle is a status 1 leaf, but nothing enforces it.
          if (reachesSim[child] != 0)
            isForest = false;
          continue;
        }
        if (simParent[child] != kNoParent) {
          isForest = false;
          continue;
        }
        simParent[child] = parent;
        simChildren[parent].push_back(child);
      }
    }
    if (!isForest)
      return false;

    std::vector<uint8_t> seen(nParticles_, 0);
    std::vector<std::pair<uint32_t, bool>> stack;

    auto visitTree = [&](uint32_t root) {
      stack.emplace_back(root, false);
      while (!stack.empty()) {
        const auto [particleId, expanded] = stack.back();
        stack.pop_back();

        if (expanded) {
          subtreeCount[particleId] = static_cast<uint32_t>(slotToParticle.size()) - dfsPos[particleId];
          continue;
        }
        if (seen[particleId] != 0)
          continue;

        seen[particleId] = 1;
        dfsPos[particleId] = static_cast<uint32_t>(slotToParticle.size());
        slotToParticle.push_back(particleId);

        // Pushed before the children, so it is popped once they have all been placed
        // and the subtree size is the slots consumed since.
        stack.emplace_back(particleId, true);
        for (const uint32_t child : simChildren[particleId]) {
          if (seen[child] == 0)
            stack.emplace_back(child, false);
        }
      }
    };

    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      if (simParent[particleId] == kNoParent)
        visitTree(particleId);
    }

    // Anything left is in a parentage cycle, which a well-formed graph does not have.
    // Giving it its own tree keeps every particle addressable.
    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      if (seen[particleId] == 0)
        visitTree(particleId);
    }

    return true;
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finish() {
    usedSharedStore_ = sharedSubgraphStore_;
    if (sharedSubgraphStore_)
      return finishShared();

    return finishMaterialised();
  }

  void LogicalGraphHitIndexBuilder::buildSubgraphRanges(std::vector<uint32_t> const& dfsPos,
                                                        std::vector<uint32_t> const& subtreeCount,
                                                        std::vector<uint32_t>& rangeOffsets,
                                                        std::vector<SlotRange>& ranges) const {
    // A particle that carries hits is a node of the SIM tree, so its subgraph is the
    // single run of slots its subtree occupies. A GEN-only particle is a node of the
    // GEN DAG above that tree: it owns no slots of its own, and its subgraph is the
    // union of the runs of the SIM particles below it, merged where they touch.
    std::vector<std::vector<SlotRange>> perParticle(nParticles_);
    std::vector<uint8_t> state(nParticles_, 0);  // 0 = new, 1 = in progress, 2 = done
    std::vector<uint32_t> stack;

    auto mergeSorted = [](std::vector<SlotRange>& runs) {
      std::sort(
          runs.begin(), runs.end(), [](SlotRange const& a, SlotRange const& b) { return a.firstSlot < b.firstSlot; });
      std::size_t w = 0;
      for (std::size_t r = 0; r < runs.size(); ++r) {
        if (w > 0 && runs[r].firstSlot <= runs[w - 1].firstSlot + runs[w - 1].slotCount) {
          const uint32_t end =
              std::max(runs[w - 1].firstSlot + runs[w - 1].slotCount, runs[r].firstSlot + runs[r].slotCount);
          runs[w - 1].slotCount = end - runs[w - 1].firstSlot;
        } else {
          runs[w++] = runs[r];
        }
      }
      runs.resize(w);
    };

    for (uint32_t seed = 0; seed < nParticles_; ++seed) {
      if (state[seed] == 2)
        continue;
      stack.push_back(seed);

      while (!stack.empty()) {
        const uint32_t particleId = stack.back();

        if (state[particleId] == 2) {
          stack.pop_back();
          continue;
        }

        if (state[particleId] == 0) {
          state[particleId] = 1;
          // A hit-carrying particle terminates the descent: its own subtree run
          // already covers everything below it.
          if (hasSimTrack_[particleId] != 0) {
            perParticle[particleId].push_back(SlotRange{dfsPos[particleId], subtreeCount[particleId]});
            state[particleId] = 2;
            stack.pop_back();
            continue;
          }
          for (const uint32_t child : children_[particleId]) {
            if (child < nParticles_ && state[child] == 0)
              stack.push_back(child);
          }
          continue;
        }

        // Second visit: every child that could be resolved has been. A child still in
        // progress is a cycle, which a well-formed graph does not have, and is skipped.
        auto& runs = perParticle[particleId];
        for (const uint32_t child : children_[particleId]) {
          if (child < nParticles_ && state[child] == 2)
            runs.insert(runs.end(), perParticle[child].begin(), perParticle[child].end());
        }
        mergeSorted(runs);
        state[particleId] = 2;
        stack.pop_back();
      }
    }

    rangeOffsets.clear();
    rangeOffsets.reserve(nParticles_ + 1);
    rangeOffsets.push_back(0);
    ranges.clear();
    for (uint32_t particleId = 0; particleId < nParticles_; ++particleId) {
      auto const& runs = perParticle[particleId];
      ranges.insert(ranges.end(), runs.begin(), runs.end());
      rangeOffsets.push_back(static_cast<uint32_t>(ranges.size()));
    }
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finishShared() {
    std::vector<uint32_t> slotToParticle;
    std::vector<uint32_t> dfsPos;
    std::vector<uint32_t> subtreeCount;
    if (!buildDfsOrder(slotToParticle, dfsPos, subtreeCount)) {
      usedSharedStore_ = false;
      return finishMaterialised();
    }

    std::vector<uint32_t> rangeOffsets;
    std::vector<SlotRange> ranges;
    buildSubgraphRanges(dfsPos, subtreeCount, rangeOffsets, ranges);

    std::vector<LogicalGraphHitIndex::Channel> channels(kNumHitChannels);

    for (std::size_t ch = 0; ch < kNumHitChannels; ++ch) {
      if (!channelTouched_[ch])
        continue;

      auto& direct = directHits_[ch];
      for (auto& hits : direct)
        coalesce(hits);

      auto& out = channels[ch];
      out.dfsOffsets.reserve(slotToParticle.size() + 1);
      out.dfsOffsets.push_back(0);

      std::size_t total = 0;
      for (auto const& hits : direct)
        total += hits.size();
      out.directHits.reserve(total);

      for (const uint32_t particleId : slotToParticle) {
        auto const& hits = direct[particleId];
        out.directHits.insert(out.directHits.end(), hits.begin(), hits.end());
        out.dfsOffsets.push_back(static_cast<uint32_t>(out.directHits.size()));
      }
    }

    return LogicalGraphHitIndex(
        nParticles_, std::move(channels), std::move(dfsPos), std::move(rangeOffsets), std::move(ranges));
  }

  LogicalGraphHitIndex LogicalGraphHitIndexBuilder::finishMaterialised() {
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
