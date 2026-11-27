// Measures the cost of one reseed followed by one draw, for each engine the
// RandomNumberGeneratorService supports. Run it pinned to one core.

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/MixMaxRng.h"
#include "FWCore/Utilities/interface/Philox.h"
#include "IOMC/RandomEngine/interface/PhiloxEngine.h"
#include "IOMC/RandomEngine/interface/TRandomAdaptor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>

namespace {
  constexpr int kCalls = 100000;
  // The largest seed of HepJamesRandom, as in the RandomNumberGeneratorService.
  constexpr long kMaxSeedHepJames = 900000000;

  double nanosecondsPerCall(std::function<double(std::uint32_t)> const& call) {
    double sink = 0.;
    auto const start = std::chrono::steady_clock::now();
    for (int i = 0; i < kCalls; ++i) {
      sink += call(static_cast<std::uint32_t>(i));
    }
    auto const stop = std::chrono::steady_clock::now();
    if (sink < 0.) {
      std::printf("impossible\n");
    }
    return std::chrono::duration<double, std::nano>(stop - start).count() / kCalls;
  }

  // Any seed and module id.
  edm::philox::Counter streamOf(std::uint32_t event) { return edm::philox::streamId(1234567, 42, 1, 1, event); }

  long streamWord(std::uint32_t event, unsigned int word) { return static_cast<long>(streamOf(event)[word]); }
}  // namespace

int main() {
  CLHEP::HepJamesRandom james(1);
  CLHEP::MixMaxRng mixMax(1);
  edm::TRandomAdaptor tRandom(1);
  edm::PhiloxEngine philox(1);

  std::printf("%-40s %12s\n", "operation (reseed + 1 flat)", "ns per call");
  std::printf("%-40s %12.1f\n", "streamId derivation only", nanosecondsPerCall([](std::uint32_t event) {
                return static_cast<double>(streamWord(event, 0));
              }));
  std::printf("%-40s %12.1f\n", "HepJamesRandom::setSeed", nanosecondsPerCall([&](std::uint32_t event) {
                james.setSeed(streamWord(event, 0) % (kMaxSeedHepJames + 1), 0);
                return james.flat();
              }));
  std::printf("%-40s %12.1f\n", "MixMaxRng::setSeed", nanosecondsPerCall([&](std::uint32_t event) {
                auto const id = streamOf(event);
                // The 64-bit seed of the service: the first two words of the stream id.
                mixMax.setSeed(static_cast<long>((static_cast<std::uint64_t>(id[1]) << 32) | id[0]), 0);
                return mixMax.flat();
              }));
  std::printf("%-40s %12.1f\n", "MixMaxRng::setSeeds(4 words)", nanosecondsPerCall([&](std::uint32_t event) {
                auto const id = streamOf(event);
                long const seeds[4] = {id[0], id[1], id[2], id[3]};
                mixMax.setSeeds(seeds, 4);
                return mixMax.flat();
              }));
  std::printf("%-40s %12.1f\n", "TRandomAdaptor::setSeed", nanosecondsPerCall([&](std::uint32_t event) {
                tRandom.setSeed(streamWord(event, 0), 0);
                return tRandom.flat();
              }));
  std::printf("%-40s %12.1f\n", "PhiloxEngine::setSeeds(4 words)", nanosecondsPerCall([&](std::uint32_t event) {
                auto const id = streamOf(event);
                long const seeds[4] = {id[0], id[1], id[2], id[3]};
                philox.setSeeds(seeds, 4);
                return philox.flat();
              }));

  std::printf("\n%-40s %12s\n", "operation", "ns per flat");
  std::printf("%-40s %12.1f\n", "MixMaxRng::flat", nanosecondsPerCall([&](std::uint32_t) { return mixMax.flat(); }));
  std::printf("%-40s %12.1f\n", "PhiloxEngine::flat", nanosecondsPerCall([&](std::uint32_t) { return philox.flat(); }));
  std::printf(
      "%-40s %12.1f\n", "HepJamesRandom::flat", nanosecondsPerCall([&](std::uint32_t) { return james.flat(); }));
  return 0;
}
