#ifndef FWCore_Utilities_Philox_h
#define FWCore_Utilities_Philox_h

// Counter-based random numbers: the Philox4x32-10 function of
// Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011.
// The functions are constexpr and have no state, so host code and device
// code (CUDA with --expt-relaxed-constexpr, HIP, Alpaka) give the same bits.

#include <array>
#include <cstdint>

namespace edm::philox {

  using Counter = std::array<std::uint32_t, 4>;
  using Key = std::array<std::uint32_t, 2>;

  constexpr std::uint32_t kMultiplier0 = 0xD2511F53;
  constexpr std::uint32_t kMultiplier1 = 0xCD9E8D57;
  constexpr std::uint32_t kWeyl0 = 0x9E3779B9;
  constexpr std::uint32_t kWeyl1 = 0xBB67AE85;
  constexpr unsigned int kRounds = 10;

  namespace detail {
    constexpr Counter round(Counter const& ctr, Key const& key) {
      std::uint64_t const product0 = static_cast<std::uint64_t>(kMultiplier0) * ctr[0];
      std::uint64_t const product1 = static_cast<std::uint64_t>(kMultiplier1) * ctr[2];
      auto const hi0 = static_cast<std::uint32_t>(product0 >> 32);
      auto const lo0 = static_cast<std::uint32_t>(product0);
      auto const hi1 = static_cast<std::uint32_t>(product1 >> 32);
      auto const lo1 = static_cast<std::uint32_t>(product1);
      return Counter{{hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0}};
    }
  }  // namespace detail

  // Philox4x32-10: a bijection of the counter for each key.
  constexpr Counter philox4x32(Counter ctr, Key key) {
    ctr = detail::round(ctr, key);
    for (unsigned int i = 1; i < kRounds; ++i) {
      key[0] += kWeyl0;
      key[1] += kWeyl1;
      ctr = detail::round(ctr, key);
    }
    return ctr;
  }

  // The 128-bit identifier of the random numbers of one (seed, module, run, lumi, event).
  // Two different events of one (seed, module) always give different identifiers.
  // Event 0 gives the stream of the lumi, lumi 0 and event 0 the stream of the run,
  // and run 0, lumi 0 and event 0 the stream of the job.
  constexpr Counter streamId(
      std::uint32_t seed, std::uint32_t moduleId, std::uint32_t run, std::uint32_t lumi, std::uint64_t event) {
    return philox4x32(Counter{{run, lumi, static_cast<std::uint32_t>(event), static_cast<std::uint32_t>(event >> 32)}},
                      Key{{seed, moduleId}});
  }

  // The random numbers of one stream identifier.
  // The object is trivially copyable: pass it to a kernel by value.
  class EventStream {
  public:
    constexpr explicit EventStream(Counter const& id) : key_{{id[0], id[1]}}, id_{{id[2], id[3]}} {}

    constexpr EventStream(
        std::uint32_t seed, std::uint32_t moduleId, std::uint32_t run, std::uint32_t lumi, std::uint64_t event)
        : EventStream(streamId(seed, moduleId, run, lumi, event)) {}

    // Four random words for draw block `block` of element `element`.
    // Each (element, block) pair gives a different counter, so the outputs do not repeat.
    constexpr Counter bits(std::uint32_t element, std::uint32_t block) const {
      return philox4x32(Counter{{block, element, id_[0], id_[1]}}, key_);
    }

  private:
    Key key_;
    std::array<std::uint32_t, 2> id_;
  };

  // Uniform in [0, 1) with 24 random bits. The conversion is exact.
  constexpr float uniformFloat(std::uint32_t word) { return static_cast<float>(word >> 8) * 0x1p-24f; }

  // Uniform in [0, 1) with 53 random bits from two words. The conversion is exact.
  constexpr double uniformDouble(std::uint32_t low, std::uint32_t high) {
    std::uint64_t const bits53 = (static_cast<std::uint64_t>(high) << 21) | (low >> 11);
    return static_cast<double>(bits53) * 0x1p-53;
  }

  // Uniform in (0, 1) with 23 random bits from one word. The conversion is exact.
  constexpr float openUniformFloat(std::uint32_t word) { return static_cast<float>(2 * (word >> 9) + 1) * 0x1p-24f; }

  // Uniform in (0, 1) with 52 random bits from two words. The conversion is exact.
  constexpr double openUniformDouble(std::uint32_t low, std::uint32_t high) {
    std::uint64_t const bits52 = ((static_cast<std::uint64_t>(high) << 32) | low) >> 12;
    return static_cast<double>(2 * bits52 + 1) * 0x1p-53;
  }

}  // namespace edm::philox

#endif
