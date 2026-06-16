// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndex_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndex_h

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace truth {

  // Detector channels of the hit index. Each channel keeps its own per-particle
  // direct hits and subgraph-aggregated hits, so calorimeter, tracker, MTD and
  // muon hits stay separate (different DetId spaces, metrics and recHit links).
  // Stored as the underlying type for dictionary simplicity; the order is the
  // channel index and must stay stable.
  enum class HitChannel : uint8_t {
    HGCalCalo = 0,  // calorimeter PCaloHits, recHit-mapped via the DetId->RecHit map
    Tracker = 1,    // tracker PSimHits, energy = energyLoss, no recHit link
    MTD = 2,        // MIP timing layer (BTL/ETL)
    Muon = 3        // muon chambers (DT/CSC/RPC/GEM)
  };
  inline constexpr std::size_t kNumHitChannels = 4;

  class LogicalGraphHitIndex {
  public:
    struct Hit {
      static constexpr uint32_t invalidRecHitIndex = std::numeric_limits<uint32_t>::max();

      uint32_t detId = 0;
      uint32_t recHitIndex = invalidRecHitIndex;
      float energy = 0.f;

      [[nodiscard]] bool hasRecHit() const { return recHitIndex != invalidRecHitIndex; }
    };

    // One detector channel: per-particle direct hits and subgraph-aggregated hits,
    // each stored CSR-style (offsets index into the flat hit storage). Empty when
    // the channel is not filled. Members are public so the dictionary and the
    // associator's flat scans can reach them directly.
    struct Channel {
      std::vector<uint32_t> directOffsets;
      std::vector<Hit> directHits;
      std::vector<uint32_t> subgraphOffsets;
      std::vector<Hit> subgraphHits;
    };

    LogicalGraphHitIndex() = default;

    LogicalGraphHitIndex(uint32_t nParticles, std::vector<Channel> channels)
        : nParticles_(nParticles), channels_(std::move(channels)) {}

    [[nodiscard]] uint32_t nParticles() const { return nParticles_; }
    [[nodiscard]] static constexpr std::size_t nChannels() { return kNumHitChannels; }

    // Direct hits of a particle in a channel (the hits on its own SimTrack).
    [[nodiscard]] std::span<const Hit> directHits(HitChannel channel, uint32_t particleId) const {
      Channel const* c = channelOrNull(channel);
      if (c == nullptr || particleId + 1 >= c->directOffsets.size())
        return {};
      const auto b = c->directOffsets[particleId];
      const auto e = c->directOffsets[particleId + 1];
      return std::span<const Hit>(c->directHits.data() + b, e - b);
    }

    // Subgraph hits of a particle in a channel (its own hits plus those of every
    // logical descendant), coalesced and sorted by detId.
    [[nodiscard]] std::span<const Hit> subgraphHits(HitChannel channel, uint32_t particleId) const {
      Channel const* c = channelOrNull(channel);
      if (c == nullptr || particleId + 1 >= c->subgraphOffsets.size())
        return {};
      const auto b = c->subgraphOffsets[particleId];
      const auto e = c->subgraphOffsets[particleId + 1];
      return std::span<const Hit>(c->subgraphHits.data() + b, e - b);
    }

    [[nodiscard]] bool hasChannel(HitChannel channel) const {
      Channel const* c = channelOrNull(channel);
      return c != nullptr && !c->directHits.empty();
    }

    // Raw channel storage (flat hit vectors + offsets), for callers that scan a
    // whole channel - e.g. BranchHitAssociator's inverted-index build.
    [[nodiscard]] Channel const& channel(HitChannel channel) const { return channels_.at(index(channel)); }

  private:
    [[nodiscard]] static constexpr std::size_t index(HitChannel channel) { return static_cast<std::size_t>(channel); }

    [[nodiscard]] Channel const* channelOrNull(HitChannel channel) const {
      const std::size_t i = index(channel);
      return i < channels_.size() ? &channels_[i] : nullptr;
    }

    uint32_t nParticles_ = 0;
    std::vector<Channel> channels_;  // size kNumHitChannels when built
  };

}  // namespace truth

#endif
