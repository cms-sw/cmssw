#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

TEST_CASE("device_unique_ptr", "[cudaMemTools]") {
  exitSansCUDADevices();

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  SECTION("Single element") {
    auto ptr = cudautils::make_device_unique<int>(stream);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Reset") {
    auto ptr = cudautils::make_device_unique<int>(stream);
    REQUIRE(ptr != nullptr);
    stream.synchronize();

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr = cudautils::make_device_unique<int[]>(10, stream);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 27; // 8**9
    auto ptr = cudautils::make_device_unique<char[]>(maxSize , stream);
    ptr.reset();
    REQUIRE_THROWS(ptr = cudautils::make_device_unique<char[]>(maxSize+1, stream));
  }
}
