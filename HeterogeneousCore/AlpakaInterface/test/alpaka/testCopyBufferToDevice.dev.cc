#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  struct Dummy {
    int x, y, z;
  };
}

TEST_CASE("Test CopyToDevice for Alpaka buffers for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  SECTION("Buffer of scalar") {
    auto buffer_host = cms::alpakatools::make_host_buffer<Dummy>();

    // run the test on each device
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      auto queue = Queue(device);
      using Copy = cms::alpakatools::CopyToDevice<decltype(buffer_host)>;
      auto buffer_device = Copy::copyAsync(queue, buffer_host);
      alpaka::wait(queue);
    }
  }

  SECTION("Buffer of array with static size") {
    // The buffer itself is really dynamically sized, even if the
    // alpakatools API looks like the array would have static size
    constexpr int N = 10;
    auto buffer_host = cms::alpakatools::make_host_buffer<int[N]>();
    for (int i = 0; i < N; ++i) {
      buffer_host[i] = i;
    }

    // run the test on each device
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      auto queue = Queue(device);
      using Copy = cms::alpakatools::CopyToDevice<decltype(buffer_host)>;
      auto buffer_device = Copy::copyAsync(queue, buffer_host);
      alpaka::wait(queue);
      REQUIRE(alpaka::getExtentProduct(buffer_device) == N);
    }
  }

  SECTION("Buffer of array with dynamic size") {
    constexpr int N = 10;
    auto buffer_host = cms::alpakatools::make_host_buffer<int[]>(N);
    for (int i = 0; i < N; ++i) {
      buffer_host[i] = i;
    }

    // run the test on each device
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      auto queue = Queue(device);
      using Copy = cms::alpakatools::CopyToDevice<decltype(buffer_host)>;
      auto buffer_device = Copy::copyAsync(queue, buffer_host);
      alpaka::wait(queue);
      REQUIRE(alpaka::getExtentProduct(buffer_device) == N);
    }
  }
}
