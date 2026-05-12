#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndex_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndex_h

#include <cstdint>
#include <span>
#include <vector>

namespace truth {

  class LogicalGraphHitIndex {
  public:
    struct Hit {
      uint32_t detId = 0;
      float energy = 0.f;
    };

    LogicalGraphHitIndex() = default;

    uint32_t nParticles() const { return nParticles_; }

    bool empty() const { return nParticles_ == 0; }

    std::span<const Hit> directHits(uint32_t particleId) const {
      return range(directOffsets_, directHits_, particleId);
    }

    std::span<const Hit> subgraphHits(uint32_t particleId) const {
      return range(subgraphOffsets_, subgraphHits_, particleId);
    }

    std::span<const Hit> totalSimHitEnergyByDetId() const { return totalSimHitEnergyByDetId_; }

    uint32_t directHitSize(uint32_t particleId) const {
      return directOffsets_.at(particleId + 1) - directOffsets_.at(particleId);
    }

    uint32_t subgraphHitSize(uint32_t particleId) const {
      return subgraphOffsets_.at(particleId + 1) - subgraphOffsets_.at(particleId);
    }

    void setData(uint32_t nParticles,
                 std::vector<uint32_t> directOffsets,
                 std::vector<Hit> directHits,
                 std::vector<uint32_t> subgraphOffsets,
                 std::vector<Hit> subgraphHits,
                 std::vector<Hit> totalSimHitEnergyByDetId) {
      nParticles_ = nParticles;
      directOffsets_ = std::move(directOffsets);
      directHits_ = std::move(directHits);
      subgraphOffsets_ = std::move(subgraphOffsets);
      subgraphHits_ = std::move(subgraphHits);
      totalSimHitEnergyByDetId_ = std::move(totalSimHitEnergyByDetId);
    }

    std::vector<uint32_t> const& directOffsets() const { return directOffsets_; }
    std::vector<Hit> const& directHitStorage() const { return directHits_; }

    std::vector<uint32_t> const& subgraphOffsets() const { return subgraphOffsets_; }
    std::vector<Hit> const& subgraphHitStorage() const { return subgraphHits_; }

  private:
    static std::span<const Hit> range(std::vector<uint32_t> const& offsets,
                                      std::vector<Hit> const& hits,
                                      uint32_t particleId) {
      const uint32_t begin = offsets.at(particleId);
      const uint32_t end = offsets.at(particleId + 1);
      return std::span<const Hit>(hits.data() + begin, end - begin);
    }

    uint32_t nParticles_ = 0;

    std::vector<uint32_t> directOffsets_;
    std::vector<Hit> directHits_;

    std::vector<uint32_t> subgraphOffsets_;
    std::vector<Hit> subgraphHits_;

    std::vector<Hit> totalSimHitEnergyByDetId_;
  };

}  // namespace truth

#endif
