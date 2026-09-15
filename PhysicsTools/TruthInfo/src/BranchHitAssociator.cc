// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

#include "DataFormats/DetId/interface/DetId.h"

namespace truth {

  uint32_t BranchHitAssociator::detectorBit(uint32_t detId) { return 1u << DetId(detId).det(); }

  void CellEnergyTable::finalize() {
    std::vector<std::pair<uint32_t, float>> cells;
    cells.reserve(keys_.size());
    for (std::size_t i = 0; i < keys_.size(); ++i)
      cells.emplace_back(keys_[i], values_[i]);
    // Rechit collections arrive sorted by DetId, and the producer adds them in
    // ascending detector order, so the common case needs no sort.
    if (!std::is_sorted(keys_.begin(), keys_.end()))
      std::sort(cells.begin(), cells.end(), [](auto const& a, auto const& b) { return a.first < b.first; });
    keys_.clear();
    values_.clear();
    for (auto const& [detId, energy] : cells) {
      if (!keys_.empty() && keys_.back() == detId)
        values_.back() += energy;
      else {
        keys_.push_back(detId);
        values_.push_back(energy);
      }
    }
  }

  float CellEnergyTable::energy(uint32_t detId) const {
    auto it = std::lower_bound(keys_.begin(), keys_.end(), detId);
    if (it == keys_.end() || *it != detId)
      return 0.f;
    return values_[static_cast<std::size_t>(it - keys_.begin())];
  }

  BranchHitAssociator::BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                           std::vector<uint32_t> candidateRoots,
                                           Metric metric,
                                           HitChannel channel,
                                           bool emptyRootsMeansAll,
                                           uint32_t denominatorDetectors,
                                           CellEnergyTable const* recHitEnergies)
      : hitIndex_(&hitIndex),
        recHitEnergies_(recHitEnergies != nullptr && !recHitEnergies->empty() ? recHitEnergies : nullptr),
        metric_(metric),
        channel_(channel),
        cellAware_(channel == HitChannel::Tracker || channel == HitChannel::Muon),
        denominatorDetectors_(denominatorDetectors),
        roots_(std::move(candidateRoots)) {
    if (roots_.empty() && emptyRootsMeansAll) {
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

    buildRootHits();

    // Inverted index detId -> candidate roots, from each candidate's subgraph
    // hits. Built as a flat (detId, root) list, sorted, then packed CSR-style so
    // lookups are a binary search plus a contiguous root span (no hashing).
    rootSelfEnergySq_.assign(hitIndex_->nParticles(), 0.0);
    rootEnergy_.assign(hitIndex_->nParticles(), 0.0);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;  // (detId, root)
    std::size_t totalRootHits = 0;
    for (const uint32_t root : roots_) {
      if (root < hitIndex_->nParticles())
        totalRootHits += rootHits(root).size();
    }
    pairs.reserve(totalRootHits);
    for (const uint32_t root : roots_) {
      if (root >= hitIndex_->nParticles())
        continue;
      double selfEnergySq = 0.0;
      double selfEnergy = 0.0;
      for (auto const& hit : rootHits(root)) {
        pairs.emplace_back(hit.detId, root);
        const float energy = branchHitEnergy(hit);
        selfEnergySq += static_cast<double>(energy) * energy;
        // The linear total is the sharedEnergyFraction denominator, so it counts only
        // the detectors the caller reconstructs; the squared one is the TICL score
        // denominator and stays over the whole channel, as the reference computes it.
        if ((denominatorDetectors_ & detectorBit(hit.detId)) != 0u)
          selfEnergy += energy;
      }
      rootSelfEnergySq_[root] = selfEnergySq;
      rootEnergy_[root] = selfEnergy;
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

  void BranchHitAssociator::buildRootHits() {
    if (!hitIndex_->sharedSubgraphStore())
      return;

    rootHitSlotOfRoot_.assign(hitIndex_->nParticles(), kNoRoot);
    rootHitOffsets_.reserve(roots_.size() + 1);
    rootHitOffsets_.push_back(0);

    std::vector<LogicalGraphHitIndex::Hit> scratch;
    for (const uint32_t root : roots_) {
      if (root >= rootHitSlotOfRoot_.size())
        continue;

      // A root whose subgraph is a single one-slot range owns nothing but its own direct
      // hits, and the builder already sorted and summed those per detId before writing
      // them, so the persisted span is usable as it stands. Skipping the copy is what
      // keeps the all-roots case affordable: leaves are most of the graph, and copying
      // every one of them would rebuild in memory the aggregate this layout exists to
      // not store.
      const auto ranges = hitIndex_->subgraphRanges(root);
      if (ranges.size() == 1 && ranges[0].slotCount == 1) {
        rootHitSlotOfRoot_[root] = kPersistedSpan;
        continue;
      }

      rootHitSlotOfRoot_[root] = static_cast<uint32_t>(rootHitOffsets_.size() - 1);

      scratch.clear();
      hitIndex_->appendSubgraphHits(channel_, root, scratch);

      // Sort by (detId, cell) and sum the energies of the entries that share that key,
      // so the merge-join sees one ascending entry per cell. On a channel whose DetId
      // already names a cell the key is the detId alone.
      std::sort(scratch.begin(), scratch.end(), [](auto const& a, auto const& b) {
        if (a.detId != b.detId)
          return a.detId < b.detId;
        return a.recHitIndex < b.recHitIndex;
      });
      std::size_t w = 0;
      for (std::size_t r = 0; r < scratch.size(); ++r) {
        const bool sameKey = w > 0 && scratch[w - 1].detId == scratch[r].detId &&
                             (!cellAware_ || scratch[w - 1].recHitIndex == scratch[r].recHitIndex);
        if (sameKey) {
          scratch[w - 1].energy += scratch[r].energy;
          if (scratch[w - 1].recHitIndex == LogicalGraphHitIndex::Hit::kInvalidRecHitIndex)
            scratch[w - 1].recHitIndex = scratch[r].recHitIndex;
        } else {
          scratch[w++] = scratch[r];
        }
      }
      scratch.resize(w);

      rootHitStorage_.insert(rootHitStorage_.end(), scratch.begin(), scratch.end());
      rootHitOffsets_.push_back(static_cast<uint32_t>(rootHitStorage_.size()));
    }
  }

  std::span<const LogicalGraphHitIndex::Hit> BranchHitAssociator::rootHits(uint32_t rootId) const {
    if (!hitIndex_->sharedSubgraphStore())
      return hitIndex_->subgraphHits(channel_, rootId);

    if (rootId >= rootHitSlotOfRoot_.size())
      return {};
    const uint32_t slot = rootHitSlotOfRoot_[rootId];
    if (slot == kNoRoot)
      return {};
    if (slot == kPersistedSpan)
      return hitIndex_->subgraphHits(channel_, rootId);
    const uint32_t begin = rootHitOffsets_[slot];
    const uint32_t end = rootHitOffsets_[slot + 1];
    return std::span<const LogicalGraphHitIndex::Hit>(rootHitStorage_.data() + begin, end - begin);
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

  float BranchHitAssociator::cellWeight(uint32_t detId) const {
    return recHitEnergies_ != nullptr ? recHitEnergies_->energy(detId) : cellTotalEnergy(detId);
  }

  float BranchHitAssociator::branchHitEnergy(LogicalGraphHitIndex::Hit const& hit) const {
    if (recHitEnergies_ == nullptr)
      return hit.energy;
    const float total = cellTotalEnergy(hit.detId);
    return total > 0.f ? hit.energy / total * recHitEnergies_->energy(hit.detId) : 0.f;
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

    // The merge-join needs the reco hits sorted the way the index stores its own, by
    // detId and then by cell. The production adapters already deliver them sorted and
    // coalesced, so the copy-and-sort only runs for an unsorted caller.
    const auto byDetId = [](RecoHit const& a, RecoHit const& b) {
      if (a.detId != b.detId)
        return a.detId < b.detId;
      return a.cell < b.cell;
    };
    std::vector<RecoHit> sortedStorage;
    std::span<const RecoHit> reco = recoHitsIn;
    if (!std::is_sorted(recoHitsIn.begin(), recoHitsIn.end(), byDetId)) {
      sortedStorage.assign(recoHitsIn.begin(), recoHitsIn.end());
      std::sort(sortedStorage.begin(), sortedStorage.end(), byDetId);
      reco = sortedStorage;
    }

    // Per-cell energy weight of the shared-energy score. The TICL trackster
    // association weights every cell by its rechit energy (squared, in the score), and
    // so does this metric when a CellEnergyTable is present: the reco object owns
    // fraction * rechit energy of a cell and the branch owns its sim fraction of the
    // same rechit energy. Without a table the weight is the cell's total sim energy
    // from the hit index, and the branch's share is its subgraph hit energy.
    const bool energyWeighted = metric_ == Metric::SharedEnergy;
    std::vector<float> cellEnergy;
    if (energyWeighted)
      cellEnergy.reserve(reco.size());

    // Self-normalization (denominator) and the set of candidate roots.
    double denominator = 0.0;
    std::vector<uint32_t> candidates;
    for (auto const& h : reco) {
      if (energyWeighted) {
        const float energy = cellWeight(h.detId);
        cellEnergy.push_back(energy);
        const double recoEnergy = static_cast<double>(h.fraction) * energy;
        denominator += recoEnergy * recoEnergy;
      }
      auto roots = rootsForCell(h.detId);
      candidates.insert(candidates.end(), roots.begin(), roots.end());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    for (const uint32_t root : candidates) {
      auto branchHits = rootHits(root);  // sorted by detId (LogicalGraphHitIndexBuilder guarantee)

      double sharedEnergy = 0.0;
      // Shared energy on the denominatorDetectors_ cells only. The fraction divides by
      // rootEnergy_, which counts those detectors, so its numerator must count the same
      // ones or a reco object with cells outside the mask exceeds 1.
      double sharedEnergyInDenominator = 0.0;
      double scoreNum = 0.0;
      // The two sides are counted separately: one reco entry can answer several branch
      // entries of a module, and one branch entry can answer several reco entries, so a
      // single counter would exceed one of the two denominators. The reco side is
      // counted per DetId run, which is one rechit, so the score is the fraction of the
      // object's rechits the branch owns whatever the cell multiplicity of a cluster.
      uint32_t recoRuns = 0;
      uint32_t sharedRecoRuns = 0;
      uint32_t sharedBranchCells = 0;

      // Branch-normalized (reverse) accumulators over the shared cells.
      double sharedBranchEnergySq = 0.0;
      double branchExcessNum = 0.0;

      // Merge-join reco hits and the branch subgraph hits by detId, then by cell where
      // the channel carries one. Both sides are sorted by (detId, cell), so the join
      // walks one DetId run at a time and each side is read once.
      std::size_t i = 0;
      std::size_t j = 0;
      while (i < reco.size()) {
        const uint32_t detId = reco[i].detId;

        std::size_t iEnd = i;
        while (iEnd < reco.size() && reco[iEnd].detId == detId)
          ++iEnd;

        while (j < branchHits.size() && branchHits[j].detId < detId)
          ++j;
        std::size_t jEnd = j;
        while (jEnd < branchHits.size() && branchHits[jEnd].detId == detId)
          ++jEnd;

        // An entry with no cell stands for the whole module and sorts last, so it
        // answers every cell the other side holds on that module.
        const bool branchHasEntry = jEnd > j;
        const bool branchCoversModule = branchHasEntry && (!cellAware_ || !branchHits[jEnd - 1].hasCell());
        bool recoNamesModuleOnly = false;
        bool moduleEntryUsed = false;
        bool runIsShared = false;
        uint32_t exactMatches = 0;
        std::size_t cursor = j;
        ++recoRuns;

        for (std::size_t r = i; r < iEnd; ++r) {
          const RecoHit& rh = reco[r];

          // The branch entry this reco entry is scored against, and whether it is the
          // first reco entry to reach it. Only the first one carries the branch-side
          // energy, which belongs to the branch entry and not to the reco entry.
          std::size_t matched = jEnd;
          bool shared = false;
          bool firstOnBranchEntry = false;
          if (!cellAware_ || rh.cell == LogicalGraphHitIndex::Hit::kNoCell) {
            recoNamesModuleOnly = branchHasEntry;
            matched = j;
            shared = branchHasEntry;
            firstOnBranchEntry = shared;
          } else {
            while (cursor < jEnd && branchHits[cursor].hasCell() && branchHits[cursor].recHitIndex < rh.cell)
              ++cursor;
            if (cursor < jEnd && branchHits[cursor].hasCell() && branchHits[cursor].recHitIndex == rh.cell) {
              matched = cursor;
              shared = true;
              firstOnBranchEntry = true;
              ++exactMatches;
              ++cursor;
            } else if (branchCoversModule) {
              matched = jEnd - 1;
              shared = true;
              firstOnBranchEntry = !moduleEntryUsed;
              moduleEntryUsed = true;
            }
          }
          runIsShared = runIsShared || shared;

          if (energyWeighted) {
            // Both directions on this cell, as the TICL association computes them:
            // the penalty is the energy the OTHER side fails to cover, squared, and an
            // excess on the other side counts as a good association rather than a
            // penalty (max(0, ...) in each direction).
            const float recoEnergy = rh.fraction * cellEnergy[r];
            const float branchEnergy = shared ? branchHitEnergy(branchHits[matched]) : 0.f;
            sharedEnergy += std::min(recoEnergy, branchEnergy);
            const float recoMinusBranch = std::max(0.f, recoEnergy - branchEnergy);
            scoreNum += static_cast<double>(recoMinusBranch) * recoMinusBranch;
            if (shared) {
              if ((denominatorDetectors_ & detectorBit(rh.detId)) != 0u)
                sharedEnergyInDenominator += std::min(recoEnergy, branchEnergy);
              if (firstOnBranchEntry) {
                sharedBranchEnergySq += static_cast<double>(branchEnergy) * branchEnergy;
                const float branchMinusReco = std::max(0.f, branchEnergy - recoEnergy);
                branchExcessNum += static_cast<double>(branchMinusReco) * branchMinusReco;
              }
            }
          }
        }

        if (runIsShared)
          ++sharedRecoRuns;

        // The branch entries this reco object answers: the exact cells, plus the
        // module-wide branch entry at most once however many reco cells fall on it. A
        // reco entry that names the module alone cannot say which cell it fired, so it
        // answers one branch entry and a branch owning more cells of the module keeps
        // the higher reverse score.
        sharedBranchCells +=
            exactMatches + (moduleEntryUsed ? 1u : 0u) + (recoNamesModuleOnly && exactMatches == 0u ? 1u : 0u);

        i = iEnd;
        j = jEnd;
      }

      if (sharedRecoRuns == 0)
        continue;

      BranchMatch m;
      m.rootParticleId = root;
      if (energyWeighted) {
        m.sharedEnergy = static_cast<float>(sharedEnergy);
        // A zero denominator means every truth-known cell of the object carries zero
        // fraction; score 1 (worst) rather than 0/0, which would put a NaN into a
        // persisted map and poison every sort and cut downstream.
        m.score = denominator > 0. ? static_cast<float>(scoreNum / denominator) : 1.f;
        // Reverse score: the fraction of the branch self-energy the reco object
        // fails to capture. Branch-only cells (not visited in the merge-join above)
        // are entirely un-captured, contributing (branchDenom - sharedBranchEnergySq)
        // to the numerator; the shared cells contribute branchExcessNum.
        const double branchDenom = rootSelfEnergySq_[root];
        const double branchScoreNum = std::max(0.0, (branchDenom - sharedBranchEnergySq) + branchExcessNum);
        // A branch with no energy of its own captures nothing, so it takes the worst
        // reverse score rather than the best one.
        m.reverseScore = branchDenom > 0.0 ? static_cast<float>(branchScoreNum / branchDenom) : 1.f;
        // Normalized to the branch energy in the detectors of denominatorDetectors_,
        // not to its whole channel energy: the numerator can only ever grow on cells
        // the reco object occupies, so a denominator spanning detectors that
        // collection does not reconstruct is a fraction nothing can pass. The numerator
        // counts the same detectors, which bounds the fraction to [0, 1].
        const double branchEnergyTotal = rootEnergy_[root];
        m.sharedEnergyFraction =
            branchEnergyTotal > 0.0 ? static_cast<float>(sharedEnergyInDenominator / branchEnergyTotal) : 0.f;
      } else {
        m.sharedEnergy = static_cast<float>(sharedRecoRuns);
        m.score = 1.f - static_cast<float>(sharedRecoRuns) / static_cast<float>(recoRuns);
        // Reverse score: fraction of the branch's cells the reco object misses.
        const std::size_t branchCellCount = branchHits.size();
        m.reverseScore = branchCellCount > 0
                             ? 1.f - static_cast<float>(sharedBranchCells) / static_cast<float>(branchCellCount)
                             : 1.f;
        m.sharedEnergyFraction =
            branchCellCount > 0 ? static_cast<float>(sharedBranchCells) / static_cast<float>(branchCellCount) : 0.f;
      }
      result.push_back(m);
    }

    // Best first. Equal scores are common, because an ancestor's subgraph holds every
    // cell of its descendants: the tightest branch, the one with the smallest reverse
    // score, then comes first, so [0] is the particle itself and not a beam proton.
    std::sort(result.begin(), result.end(), [](BranchMatch const& a, BranchMatch const& b) {
      if (a.score != b.score)
        return a.score < b.score;
      if (a.reverseScore != b.reverseScore)
        return a.reverseScore < b.reverseScore;
      return a.rootParticleId < b.rootParticleId;
    });

    if (maxResults > 0 && result.size() > maxResults)
      result.resize(maxResults);

    return result;
  }

  BranchMatch BranchHitAssociator::bestAdaptiveBranch(std::span<const RecoHit> recoHits,
                                                      float reverseWeight,
                                                      float maxReverseScore) const {
    // Reuse the full merge-join (both scores per candidate) and re-rank by the
    // balanced objective. The candidate set already encodes the climb: with the
    // ancestor closure as roots, a reco hit lands on its leaf and every ancestor,
    // so all levels appear here and the argmin selects the best one.
    const auto all = bestBranches(recoHits, 0);
    return bestAdaptiveBranch(all, reverseWeight, maxReverseScore);
  }

  BranchMatch BranchHitAssociator::bestAdaptiveBranch(std::span<const BranchMatch> matches,
                                                      float reverseWeight,
                                                      float maxReverseScore) {
    // Float throughout: every input is a float map payload and nothing accumulates,
    // so double buys no precision here. One pass tracks the ceiling-constrained and
    // the unconstrained argmin together; the fallback to the unconstrained minimum
    // covers a ceiling that rejected every level (e.g. a very fragmented reco
    // object), which must not be reported as no match.
    const auto objective = [reverseWeight](BranchMatch const& m) { return m.score + reverseWeight * m.reverseScore; };

    BranchMatch best;
    best.rootParticleId = BranchMatch::kInvalidRoot;
    float bestObj = std::numeric_limits<float>::infinity();
    BranchMatch fallback;
    fallback.rootParticleId = BranchMatch::kInvalidRoot;
    float fallbackObj = std::numeric_limits<float>::infinity();
    for (auto const& m : matches) {
      const float obj = objective(m);
      if (m.reverseScore <= maxReverseScore && obj < bestObj) {
        bestObj = obj;
        best = m;
      }
      if (obj < fallbackObj) {
        fallbackObj = obj;
        fallback = m;
      }
    }
    return best.rootParticleId != BranchMatch::kInvalidRoot ? best : fallback;
  }

}  // namespace truth
