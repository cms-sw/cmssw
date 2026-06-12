#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  class LogicalGraphHitIndexBuilder {
  public:
    explicit LogicalGraphHitIndexBuilder(uint32_t nParticles);

    void setSimTrackForParticle(uint32_t particleId, uint32_t trackId);
    void addParticleChild(uint32_t parentParticleId, uint32_t childParticleId);

    void addHitForTrack(uint32_t trackId, uint32_t detId, uint32_t recHitIndex, float energy);

    // Tracker simhits: separate channel, no recHit association.
    void addTrackerHitForTrack(uint32_t trackId, uint32_t detId, float energy);

    [[nodiscard]] LogicalGraphHitIndex finish();

  private:
    using Hit = LogicalGraphHitIndex::Hit;

    struct HitAccumulator {
      uint32_t recHitIndex = Hit::invalidRecHitIndex;
      float energy = 0.f;
    };

    using HitMap = std::unordered_map<uint32_t, HitAccumulator>;

    static void addHit(HitMap& hits, uint32_t detId, uint32_t recHitIndex, float energy);
    static void addHit(HitMap& hits, Hit const& hit);
    static std::vector<Hit> sortedHits(HitMap const& hits);

    // Aggregate direct hits of a particle and all its descendants into subgraph.
    void fillSubgraphHits(uint32_t particleId,
                          std::vector<HitMap> const& direct,
                          std::vector<HitMap>& subgraph,
                          std::vector<uint8_t>& state);

    static void buildHitCSR(std::vector<HitMap> const& maps, std::vector<uint32_t>& offsets, std::vector<Hit>& storage);

    uint32_t nParticles_ = 0;

    std::unordered_map<uint32_t, uint32_t> trackIdToParticle_;
    std::vector<std::vector<uint32_t>> children_;

    std::vector<HitMap> directHits_;
    std::vector<HitMap> subgraphHits_;

    std::vector<HitMap> trackerDirectHits_;
    std::vector<HitMap> trackerSubgraphHits_;
  };

}  // namespace truth

#endif
