#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  class LogicalGraphHitIndexBuilder {
  public:
    using Hit = LogicalGraphHitIndex::Hit;

    LogicalGraphHitIndexBuilder() = default;

    explicit LogicalGraphHitIndexBuilder(uint32_t nParticles) { reset(nParticles); }

    void reset(uint32_t nParticles);

    uint32_t nParticles() const { return nParticles_; }

    void setSimTrackForParticle(uint32_t particleId, uint32_t trackId);

    void addParticleChild(uint32_t parentParticleId, uint32_t childParticleId);

    void addHitForTrack(uint32_t trackId, uint32_t detId, float energy);

    void addHitForParticle(uint32_t particleId, uint32_t detId, float energy);

    void sortAndReduceDirectHits();

    LogicalGraphHitIndex finish();

    std::vector<Hit> collectMergedDirectHits(std::span<const uint32_t> particles) const;

    std::vector<Hit> collectMergedSubgraphHits(std::span<const uint32_t> particles) const;

  private:
    static void sortAndReduce(std::vector<Hit>& hits);

    static std::vector<Hit> mergeSortedHitLists(std::span<const Hit> a, std::span<const Hit> b);

    static void mergeInto(std::vector<Hit>& dst, std::span<const Hit> src);

    void computeSubgraphHits();

    void buildCsr(std::vector<std::vector<Hit>> const& perParticleHits,
                  std::vector<uint32_t>& offsets,
                  std::vector<Hit>& storage) const;

    void collectSubgraphParticles(uint32_t rootParticleId, std::vector<uint8_t>& visited) const;

    uint32_t nParticles_ = 0;

    std::unordered_map<uint32_t, uint32_t> trackIdToParticle_;

    std::vector<std::vector<uint32_t>> children_;

    std::vector<std::vector<Hit>> directHits_;
    std::vector<std::vector<Hit>> subgraphHits_;

    std::vector<Hit> totalSimHitEnergyByDetId_;
  };

}  // namespace truth

#endif
