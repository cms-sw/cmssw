#include <cstdint>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/atomicIncSaturate.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestAtomicIncSaturate) "]";

// Each element increments the counter, and records which index it was given:
//   - indices below the limit are counted in the histogram, and must be given out exactly once;
//   - all other calls must return the limit itself, and are counted in the overflow counter.
struct IncrementKernel {
  template <typename T>
  ALPAKA_FN_ACC void operator()(
      Acc1D const& acc, T* counter, T limit, uint32_t* histogram, uint32_t* overflow, uint32_t size) const {
    for ([[maybe_unused]] uint32_t i : uniform_elements(acc, size)) {
      T index = atomicIncSaturate(acc, counter, limit, alpaka::hierarchy::Blocks{});
      if (index < limit) {
        alpaka::atomicAdd(acc, &histogram[index], 1u, alpaka::hierarchy::Blocks{});
      } else {
        ALPAKA_ASSERT_ACC(index == limit);
        alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  }
};

template <typename T>
void testAtomicIncSaturate(Queue& queue, uint32_t size, T limit) {
  // The histogram covers all the indices that may be given out.
  uint32_t bins = std::min(size, static_cast<uint32_t>(limit));
  uint32_t expected = std::min(size, static_cast<uint32_t>(limit));

  auto counter_d = make_device_buffer<T>(queue);
  auto overflow_d = make_device_buffer<uint32_t>(queue);
  auto histogram_d = make_device_buffer<uint32_t[]>(queue, bins);
  alpaka::memset(queue, counter_d, 0x00);
  alpaka::memset(queue, overflow_d, 0x00);
  alpaka::memset(queue, histogram_d, 0x00);

  auto workdiv = make_workdiv<Acc1D>(divide_up_by(size, 256u), 256u);
  alpaka::exec<Acc1D>(
      queue, workdiv, IncrementKernel{}, counter_d.data(), limit, histogram_d.data(), overflow_d.data(), size);

  auto counter_h = make_host_buffer<T>(queue);
  auto overflow_h = make_host_buffer<uint32_t>(queue);
  auto histogram_h = make_host_buffer<uint32_t[]>(queue, bins);
  alpaka::memcpy(queue, counter_h, counter_d);
  alpaka::memcpy(queue, overflow_h, overflow_d);
  alpaka::memcpy(queue, histogram_h, histogram_d);
  alpaka::wait(queue);

  // The counter saturates at the limit.
  REQUIRE(static_cast<uint32_t>(*counter_h) == expected);
  // All calls past the limit are reported as overflows.
  REQUIRE(*overflow_h == size - expected);
  // Each index below the limit is given out exactly once.
  for (uint32_t i = 0; i < bins; ++i) {
    REQUIRE(histogram_h[i] == 1u);
  }
}

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestAtomicIncSaturate), s_tag) {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
         "the test will be skipped.");
  }

  // run the test on each device
  for (auto const& device : devices) {
    std::cout << "Test atomicIncSaturate on " << alpaka::getName(device) << '\n';
    Queue queue(device);

    SECTION("int32_t, saturating") { testAtomicIncSaturate<int32_t>(queue, 100000, 1000); }
    SECTION("int32_t, not saturating") { testAtomicIncSaturate<int32_t>(queue, 1000, 100000); }
    SECTION("int32_t, exactly at the limit") { testAtomicIncSaturate<int32_t>(queue, 1000, 1000); }
    SECTION("int32_t, zero limit") { testAtomicIncSaturate<int32_t>(queue, 1000, 0); }
    SECTION("uint32_t, saturating") { testAtomicIncSaturate<uint32_t>(queue, 100000, 1000); }
    SECTION("uint32_t, not saturating") { testAtomicIncSaturate<uint32_t>(queue, 1000, 100000); }
  }
}
