// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>

namespace truth {

  BranchHitAssociator::BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                           std::vector<uint32_t> candidateRoots,
                                           Metric metric,
                                           HitChannel channel)
      : hitIndex_(&hitIndex), metric_(metric), channel_(channel), roots_(std::move(candidateRoots)) {
    if (roots_.empty()) {
      roots_.resize(hitIndex_->nParticles());
      std::iota(roots_.begin(), roots_.end(), 0u);
    }

    // Per-cell total sim energy (denominator for branch fractions): sum of every
    // particle's direct-hit energy on that cell. Use the requested channel.
    // directStorage is grouped by particle, not globally sorted, so collect and
    // coalesce into a sorted (detId -> energy) table for binary-search lookup.
    const auto& directStorage = hitIndex_->channel(channel_).directHits;
    std::vector<std::pair<uint32_t, float>> cells;
    cells.reserve(directStorage.size());
    for (auto const& hit : directStorage)
      cells.emplace_back(hit.detId, hit.energy);
    std::sort(cells.begin(), cells.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

    cellEnergyKeys_.reserve(cells.size());
    cellEnergyValues_.reserve(cells.size());
    for (auto const& [detId, energy] : cells) {
      if (!cellEnergyKeys_.empty() && cellEnergyKeys_.back() == detId)
        cellEnergyValues_.back() += energy;
      else {
        cellEnergyKeys_.push_back(detId);
        cellEnergyValues_.push_back(energy);
      }
    }

    // Inverted index detId -> candidate roots, from each candidate's subgraph
    // hits. Built as a flat (detId, root) list, sorted, then packed CSR-style so
    // lookups are a binary search plus a contiguous root span (no hashing).
    std::vector<std::pair<uint32_t, uint32_t>> pairs;  // (detId, root)
    for (const uint32_t root : roots_) {
      if (root >= hitIndex_->nParticles())
        continue;
      for (auto const& hit : rootHits(root))
        pairs.emplace_back(hit.detId, root);
    }
    std::sort(pairs.begin(), pairs.end());  // by detId, then root

    cellRootsOffsets_.push_back(0);
    cellRoots_.reserve(pairs.size());
    for (std::size_t i = 0; i < pairs.size();) {
      const uint32_t detId = pairs[i].first;
      cellRootsKeys_.push_back(detId);
      std::size_t j = i;
      while (j < pairs.size() && pairs[j].first == detId) {
        cellRoots_.push_back(pairs[j].second);
        ++j;
      }
      cellRootsOffsets_.push_back(static_cast<uint32_t>(cellRoots_.size()));
      i = j;
    }
  }

  std::span<const LogicalGraphHitIndex::Hit> BranchHitAssociator::rootHits(uint32_t rootId) const {
    return hitIndex_->subgraphHits(channel_, rootId);
  }

  std::span<const uint32_t> BranchHitAssociator::rootsForCell(uint32_t detId) const {
    auto it = std::lower_bound(cellRootsKeys_.begin(), cellRootsKeys_.end(), detId);
    if (it == cellRootsKeys_.end() || *it != detId)
      return {};
    const std::size_t k = static_cast<std::size_t>(it - cellRootsKeys_.begin());
    const uint32_t b = cellRootsOffsets_[k];
    const uint32_t e = cellRootsOffsets_[k + 1];
    return std::span<const uint32_t>(cellRoots_.data() + b, e - b);
  }

  float BranchHitAssociator::cellTotalEnergy(uint32_t detId) const {
    auto it = std::lower_bound(cellEnergyKeys_.begin(), cellEnergyKeys_.end(), detId);
    if (it == cellEnergyKeys_.end() || *it != detId)
      return 0.f;
    return cellEnergyValues_[static_cast<std::size_t>(it - cellEnergyKeys_.begin())];
  }

  std::vector<BranchMatch> BranchHitAssociator::bestBranches(std::span<const RecoHit> recoHitsIn,
                                                             std::size_t maxResults) const {
    std::vector<BranchMatch> result;
    if (recoHitsIn.empty())
      return result;

    // Sort the reco object's hits by detId for the merge-join.
    std::vector<RecoHit> reco(recoHitsIn.begin(), recoHitsIn.end());
    std::sort(reco.begin(), reco.end(), [](RecoHit const& a, RecoHit const& b) { return a.detId < b.detId; });

    // Self-normalization (denominator) and the set of candidate roots.
    double denominator = 0.0;
    std::vector<uint32_t> candidates;
    for (auto const& h : reco) {
      denominator += static_cast<double>(h.fraction * h.energy) * (h.fraction * h.energy);
      auto roots = rootsForCell(h.detId);
      candidates.insert(candidates.end(), roots.begin(), roots.end());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (denominator <= 0.0)
      denominator = 1.0;

    for (const uint32_t root : candidates) {
      auto branchHits = rootHits(root);  // sorted by detId (LogicalGraphHitIndexBuilder guarantee)

      double sharedEnergy = 0.0;
      double scoreNum = 0.0;
      uint32_t sharedCells = 0;

      // Merge-join reco hits and the branch subgraph hits by detId.
      std::size_t i = 0;
      std::size_t j = 0;
      while (i < reco.size()) {
        const RecoHit& rh = reco[i];

        // advance branch pointer to rh.detId
        while (j < branchHits.size() && branchHits[j].detId < rh.detId)
          ++j;

        float branchFraction = 0.f;
        if (j < branchHits.size() && branchHits[j].detId == rh.detId) {
          const float cellTotal = cellTotalEnergy(rh.detId);
          branchFraction = cellTotal > 0.f ? branchHits[j].energy / cellTotal : 0.f;
          ++sharedCells;
        }

        if (metric_ == Metric::SharedEnergy) {
          sharedEnergy += std::min(branchFraction * rh.energy, rh.fraction * rh.energy);
          const float excess = std::max(0.f, rh.fraction - branchFraction);
          scoreNum += static_cast<double>(excess * rh.energy) * (excess * rh.energy);
        }
        ++i;
      }

      BranchMatch m;
      m.rootParticleId = root;
      if (metric_ == Metric::SharedEnergy) {
        if (sharedCells == 0)
          continue;
        m.sharedEnergy = static_cast<float>(sharedEnergy);
        m.score = static_cast<float>(scoreNum / denominator);
      } else {
        if (sharedCells == 0)
          continue;
        m.sharedEnergy = static_cast<float>(sharedCells);
        m.score = 1.f - static_cast<float>(sharedCells) / static_cast<float>(reco.size());
      }
      result.push_back(m);
    }

    std::sort(result.begin(), result.end(), [](BranchMatch const& a, BranchMatch const& b) {
      return a.score != b.score ? a.score < b.score : a.rootParticleId < b.rootParticleId;
    });

    if (maxResults > 0 && result.size() > maxResults)
      result.resize(maxResults);

    return result;
  }

}  // namespace truth
