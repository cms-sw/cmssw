#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"

#include <algorithm>
#include <numeric>

namespace truth {

  BranchHitAssociator::BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                           std::vector<uint32_t> candidateRoots,
                                           Metric metric,
                                           bool useTracker)
      : hitIndex_(&hitIndex), metric_(metric), useTracker_(useTracker), roots_(std::move(candidateRoots)) {
    if (roots_.empty()) {
      roots_.resize(hitIndex_->nParticles());
      std::iota(roots_.begin(), roots_.end(), 0u);
    }

    // Per-cell total sim energy (denominator for branch fractions): sum of every
    // particle's direct-hit energy on that cell. Use the requested channel.
    const auto& directStorage = useTracker_ ? hitIndex_->trackerDirectHitStorage() : hitIndex_->directHitStorage();
    for (auto const& hit : directStorage)
      cellTotalEnergy_[hit.detId] += hit.energy;

    // Inverted index detId -> candidate roots, from each candidate's subgraph hits.
    for (const uint32_t root : roots_) {
      if (root >= hitIndex_->nParticles())
        continue;
      for (auto const& hit : rootHits(root))
        detIdToRoots_[hit.detId].push_back(root);
    }
  }

  std::span<const LogicalGraphHitIndex::Hit> BranchHitAssociator::rootHits(uint32_t rootId) const {
    return useTracker_ ? hitIndex_->trackerSubgraphHits(rootId) : hitIndex_->subgraphHits(rootId);
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
      auto it = detIdToRoots_.find(h.detId);
      if (it != detIdToRoots_.end())
        candidates.insert(candidates.end(), it->second.begin(), it->second.end());
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
          const float cellTotal = cellTotalEnergy_.count(rh.detId) ? cellTotalEnergy_.at(rh.detId) : 0.f;
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
