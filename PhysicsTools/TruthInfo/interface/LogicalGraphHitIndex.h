// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndex_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndex_h

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace truth {

  class LogicalGraphHitIndex {
  public:
    struct Hit {
      static constexpr uint32_t invalidRecHitIndex = std::numeric_limits<uint32_t>::max();

      uint32_t detId = 0;
      uint32_t recHitIndex = invalidRecHitIndex;
      float energy = 0.f;

      [[nodiscard]] bool hasRecHit() const { return recHitIndex != invalidRecHitIndex; }
    };

    LogicalGraphHitIndex() = default;

    LogicalGraphHitIndex(uint32_t nParticles,
                         std::vector<uint32_t> directOffsets,
                         std::vector<Hit> directHits,
                         std::vector<uint32_t> subgraphOffsets,
                         std::vector<Hit> subgraphHits,
                         std::vector<uint32_t> trackerDirectOffsets = {},
                         std::vector<Hit> trackerDirectHits = {},
                         std::vector<uint32_t> trackerSubgraphOffsets = {},
                         std::vector<Hit> trackerSubgraphHits = {})
        : nParticles_(nParticles),
          directOffsets_(std::move(directOffsets)),
          directHits_(std::move(directHits)),
          subgraphOffsets_(std::move(subgraphOffsets)),
          subgraphHits_(std::move(subgraphHits)),
          trackerDirectOffsets_(std::move(trackerDirectOffsets)),
          trackerDirectHits_(std::move(trackerDirectHits)),
          trackerSubgraphOffsets_(std::move(trackerSubgraphOffsets)),
          trackerSubgraphHits_(std::move(trackerSubgraphHits)) {}

    [[nodiscard]] uint32_t nParticles() const { return nParticles_; }

    [[nodiscard]] std::span<const Hit> directHits(uint32_t particleId) const {
      const auto b = directOffsets_.at(particleId);
      const auto e = directOffsets_.at(particleId + 1);
      return std::span<const Hit>(directHits_.data() + b, e - b);
    }

    [[nodiscard]] std::span<const Hit> subgraphHits(uint32_t particleId) const {
      const auto b = subgraphOffsets_.at(particleId);
      const auto e = subgraphOffsets_.at(particleId + 1);
      return std::span<const Hit>(subgraphHits_.data() + b, e - b);
    }

    [[nodiscard]] const std::vector<uint32_t>& directOffsets() const { return directOffsets_; }
    [[nodiscard]] const std::vector<Hit>& directHitStorage() const { return directHits_; }

    [[nodiscard]] const std::vector<uint32_t>& subgraphOffsets() const { return subgraphOffsets_; }
    [[nodiscard]] const std::vector<Hit>& subgraphHitStorage() const { return subgraphHits_; }

    // Tracker simhits are kept in a separate channel. They carry no recHit
    // association (recHitIndex stays invalid) and their energy is the PSimHit
    // energy loss. Spans are empty when tracker matching is not configured.
    [[nodiscard]] std::span<const Hit> trackerDirectHits(uint32_t particleId) const {
      if (particleId + 1 >= trackerDirectOffsets_.size())
        return {};
      const auto b = trackerDirectOffsets_[particleId];
      const auto e = trackerDirectOffsets_[particleId + 1];
      return std::span<const Hit>(trackerDirectHits_.data() + b, e - b);
    }

    [[nodiscard]] std::span<const Hit> trackerSubgraphHits(uint32_t particleId) const {
      if (particleId + 1 >= trackerSubgraphOffsets_.size())
        return {};
      const auto b = trackerSubgraphOffsets_[particleId];
      const auto e = trackerSubgraphOffsets_[particleId + 1];
      return std::span<const Hit>(trackerSubgraphHits_.data() + b, e - b);
    }

    [[nodiscard]] bool hasTrackerHits() const { return !trackerDirectHits_.empty(); }

    [[nodiscard]] const std::vector<Hit>& trackerDirectHitStorage() const { return trackerDirectHits_; }
    [[nodiscard]] const std::vector<Hit>& trackerSubgraphHitStorage() const { return trackerSubgraphHits_; }

  private:
    uint32_t nParticles_ = 0;

    std::vector<uint32_t> directOffsets_;
    std::vector<Hit> directHits_;

    std::vector<uint32_t> subgraphOffsets_;
    std::vector<Hit> subgraphHits_;

    std::vector<uint32_t> trackerDirectOffsets_;
    std::vector<Hit> trackerDirectHits_;

    std::vector<uint32_t> trackerSubgraphOffsets_;
    std::vector<Hit> trackerSubgraphHits_;
  };

}  // namespace truth

#endif
