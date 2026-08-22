#include <cstddef>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

// The allocator is exercised on the host device, so the test needs no accelerator.
using Device = alpaka::DevCpu;
using Queue = alpaka::QueueCpuBlocking;
using Allocator = cms::alpakatools::CachingAllocator<Device, Queue>;

TEST_CASE("A reused block reports the size of the new allocation", "[CachingAllocator]") {
  auto const& device = cms::alpakatools::host();
  cms::alpakatools::AllocatorConfig config;
  // reuse blocks that are associated to the same queue, as the device allocators do
  Allocator allocator{device, config, true, false};
  Queue queue{device};

  // 300 and 500 bytes fall in the same bin, so the second allocation reuses the first block
  void* first = allocator.allocate(300, queue);
  REQUIRE(allocator.cacheStatus().requested == 300);
  REQUIRE(allocator.cacheStatus().live == 512);

  allocator.free(first);
  REQUIRE(allocator.cacheStatus().requested == 0);
  REQUIRE(allocator.cacheStatus().live == 0);
  REQUIRE(allocator.cacheStatus().free == 512);

  void* second = allocator.allocate(500, queue);
  REQUIRE(second == first);
  REQUIRE(allocator.cacheStatus().live == 512);
  // the block is the same, but the reported size is the one of this allocation
  REQUIRE(allocator.cacheStatus().requested == 500);

  allocator.free(second);
  REQUIRE(allocator.cacheStatus().requested == 0);
}

TEST_CASE("The accounting returns to zero after many reuses", "[CachingAllocator]") {
  auto const& device = cms::alpakatools::host();
  cms::alpakatools::AllocatorConfig config;
  Allocator allocator{device, config, true, false};
  Queue queue{device};

  for (size_t bytes : {300ul, 500ul, 260ul, 511ul, 300ul}) {
    void* p = allocator.allocate(bytes, queue);
    REQUIRE(allocator.cacheStatus().requested == bytes);
    allocator.free(p);
    REQUIRE(allocator.cacheStatus().requested == 0);
    REQUIRE(allocator.cacheStatus().live == 0);
  }
}
