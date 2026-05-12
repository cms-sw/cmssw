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
                         std::vector<Hit> subgraphHits)
        : nParticles_(nParticles),
          directOffsets_(std::move(directOffsets)),
          directHits_(std::move(directHits)),
          subgraphOffsets_(std::move(subgraphOffsets)),
          subgraphHits_(std::move(subgraphHits)) {}

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

  private:
    uint32_t nParticles_ = 0;

    std::vector<uint32_t> directOffsets_;
    std::vector<Hit> directHits_;

    std::vector<uint32_t> subgraphOffsets_;
    std::vector<Hit> subgraphHits_;
  };

}  // namespace truth

#endif
