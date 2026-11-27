#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/Philox.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/random.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  // 2^18 elements x 4 blocks x 4 words = 4.2 million words.
  constexpr std::uint32_t kElements = 1 << 18;
  constexpr std::uint32_t kBlocks = 4;
  constexpr std::uint32_t kWordsPerBlock = 4;
  constexpr std::uint32_t kWords = kElements * kBlocks * kWordsPerBlock;
  // The output does not depend on the block size.
  constexpr std::uint32_t kThreadsPerBlock = 256;
  // The math functions of a GPU backend can differ from the host ones in the last bits.
  // CUDA gives at most 3 ulp; the limit leaves a margin for other GPUs.
  constexpr std::int64_t kMaxNormalUlpGpu = 8;
  // |value| <= sqrt(-2 ln 2^-32) = 6.66, see normalFloat4.
  constexpr float kMaxNormal = 6.67f;

  struct PhiloxKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  edm::philox::EventStream stream,
                                  std::uint32_t* __restrict__ words,
                                  float* __restrict__ floats,
                                  double* __restrict__ doubles,
                                  float* __restrict__ openFloats,
                                  double* __restrict__ openDoubles,
                                  float* __restrict__ normals) const {
      for (auto element : cms::alpakatools::uniform_elements(acc, kElements)) {
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
          auto const bits = stream.bits(element, block);
          auto const normal = cms::alpakatools::normalFloat4(acc, bits);
          auto const first = (element * kBlocks + block) * kWordsPerBlock;
          for (std::uint32_t word = 0; word < kWordsPerBlock; ++word) {
            words[first + word] = bits[word];
            floats[first + word] = edm::philox::uniformFloat(bits[word]);
            openFloats[first + word] = edm::philox::openUniformFloat(bits[word]);
            normals[first + word] = normal[word];
          }
          doubles[first / 2] = edm::philox::uniformDouble(bits[0], bits[1]);
          doubles[first / 2 + 1] = edm::philox::uniformDouble(bits[2], bits[3]);
          openDoubles[first / 2] = edm::philox::openUniformDouble(bits[0], bits[1]);
          openDoubles[first / 2 + 1] = edm::philox::openUniformDouble(bits[2], bits[3]);
        }
      }
    }
  };

  // Host reference of cms::alpakatools::normalFloat4 with the standard library math functions.
  std::array<float, 4> normalFloat4Host(edm::philox::Counter const& bits) {
    std::array<float, 4> values;
    for (unsigned int pair = 0; pair < 2; ++pair) {
      float const u1 = (static_cast<float>(bits[2 * pair]) + 1.f) * 0x1p-32f;
      float const u2 = edm::philox::uniformFloat(bits[2 * pair + 1]);
      float const radius = std::sqrt(-2.f * std::log(u1));
      float const angle = 2.f * std::numbers::pi_v<float> * u2;
      values[2 * pair] = radius * std::cos(angle);
      values[2 * pair + 1] = radius * std::sin(angle);
    }
    return values;
  }

  // Distance in units in the last place between two finite floats.
  std::int64_t ulpDistance(float a, float b) {
    auto ordered = [](float x) {
      auto const i = std::bit_cast<std::int32_t>(x);
      return i < 0 ? static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) - i
                   : static_cast<std::int64_t>(i);
    };
    return std::abs(ordered(a) - ordered(b));
  }
}  // namespace

TEST_CASE("Philox random numbers on the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
         "the test will be skipped.");
  }

  SECTION("device output equals host output") {
    // Any seed, module, run and lumi. The event number is above 2^32, so both event words are used.
    edm::philox::EventStream const stream(1234567, 42, 1, 7, (std::uint64_t{1} << 40) + 12345);

    auto words_h = cms::alpakatools::make_host_buffer<std::uint32_t[], Platform>(kWords);
    auto floats_h = cms::alpakatools::make_host_buffer<float[], Platform>(kWords);
    auto doubles_h = cms::alpakatools::make_host_buffer<double[], Platform>(kWords / 2);
    auto openFloats_h = cms::alpakatools::make_host_buffer<float[], Platform>(kWords);
    auto openDoubles_h = cms::alpakatools::make_host_buffer<double[], Platform>(kWords / 2);
    auto normals_h = cms::alpakatools::make_host_buffer<float[], Platform>(kWords);

    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      std::printf("Test Philox on %s\n", alpaka::getName(device).c_str());
      auto queue = Queue(device);
      auto words_d = cms::alpakatools::make_device_buffer<std::uint32_t[]>(queue, kWords);
      auto floats_d = cms::alpakatools::make_device_buffer<float[]>(queue, kWords);
      auto doubles_d = cms::alpakatools::make_device_buffer<double[]>(queue, kWords / 2);
      auto openFloats_d = cms::alpakatools::make_device_buffer<float[]>(queue, kWords);
      auto openDoubles_d = cms::alpakatools::make_device_buffer<double[]>(queue, kWords / 2);
      auto normals_d = cms::alpakatools::make_device_buffer<float[]>(queue, kWords);

      auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(
          cms::alpakatools::divide_up_by(kElements, kThreadsPerBlock), kThreadsPerBlock);
      alpaka::exec<Acc1D>(queue,
                          workDiv,
                          PhiloxKernel{},
                          stream,
                          words_d.data(),
                          floats_d.data(),
                          doubles_d.data(),
                          openFloats_d.data(),
                          openDoubles_d.data(),
                          normals_d.data());
      alpaka::memcpy(queue, words_h, words_d);
      alpaka::memcpy(queue, floats_h, floats_d);
      alpaka::memcpy(queue, doubles_h, doubles_d);
      alpaka::memcpy(queue, openFloats_h, openFloats_d);
      alpaka::memcpy(queue, openDoubles_h, openDoubles_d);
      alpaka::memcpy(queue, normals_h, normals_d);
      alpaka::wait(queue);

      std::uint32_t differentWords = 0;
      std::uint32_t differentFloats = 0;
      std::uint32_t differentDoubles = 0;
      std::int64_t maxNormalUlp = 0;
      double normalSum = 0.;
      double normalSum2 = 0.;
      float normalMaxAbs = 0.f;
      for (std::uint32_t element = 0; element < kElements; ++element) {
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
          auto const bits = stream.bits(element, block);
          auto const normal = normalFloat4Host(bits);
          auto const first = (element * kBlocks + block) * kWordsPerBlock;
          for (std::uint32_t word = 0; word < kWordsPerBlock; ++word) {
            differentWords += words_h[first + word] != bits[word];
            differentFloats += std::bit_cast<std::uint32_t>(floats_h[first + word]) !=
                               std::bit_cast<std::uint32_t>(edm::philox::uniformFloat(bits[word]));
            differentFloats += std::bit_cast<std::uint32_t>(openFloats_h[first + word]) !=
                               std::bit_cast<std::uint32_t>(edm::philox::openUniformFloat(bits[word]));
            float const deviceNormal = normals_h[first + word];
            normalSum += deviceNormal;
            normalSum2 += static_cast<double>(deviceNormal) * deviceNormal;
            normalMaxAbs = std::isfinite(deviceNormal) ? std::max(normalMaxAbs, std::abs(deviceNormal))
                                                       : std::numeric_limits<float>::infinity();
            maxNormalUlp = std::max(maxNormalUlp, ulpDistance(normals_h[first + word], normal[word]));
          }
          differentDoubles += std::bit_cast<std::uint64_t>(doubles_h[first / 2]) !=
                              std::bit_cast<std::uint64_t>(edm::philox::uniformDouble(bits[0], bits[1]));
          differentDoubles += std::bit_cast<std::uint64_t>(doubles_h[first / 2 + 1]) !=
                              std::bit_cast<std::uint64_t>(edm::philox::uniformDouble(bits[2], bits[3]));
          differentDoubles += std::bit_cast<std::uint64_t>(openDoubles_h[first / 2]) !=
                              std::bit_cast<std::uint64_t>(edm::philox::openUniformDouble(bits[0], bits[1]));
          differentDoubles += std::bit_cast<std::uint64_t>(openDoubles_h[first / 2 + 1]) !=
                              std::bit_cast<std::uint64_t>(edm::philox::openUniformDouble(bits[2], bits[3]));
        }
      }
      std::printf("%u words, %u floats, %u doubles differ; maximum normal difference %lld ulp\n",
                  differentWords,
                  differentFloats,
                  differentDoubles,
                  static_cast<long long>(maxNormalUlp));
      CHECK(differentWords == 0);
      CHECK(differentFloats == 0);
      CHECK(differentDoubles == 0);
      if constexpr (std::is_same_v<Platform, alpaka::PlatformCpu>) {
        CHECK(maxNormalUlp == 0);
      } else {
        CHECK(maxNormalUlp <= kMaxNormalUlpGpu);
      }
      // Mean 0 and variance 1 within 5 standard deviations.
      // For N normal values, the standard deviation is 1/sqrt(N) for the mean and sqrt(2/N) for the variance.
      double const mean = normalSum / kWords;
      double const variance = normalSum2 / kWords - mean * mean;
      std::printf("normal mean %.6f, variance %.6f, maximum |value| %.4f\n", mean, variance, normalMaxAbs);
      CHECK(std::abs(mean) < 5. / std::sqrt(kWords));
      CHECK(std::abs(variance - 1.) < 5. * std::sqrt(2. / kWords));
      CHECK(normalMaxAbs < kMaxNormal);
    }
  }
}
