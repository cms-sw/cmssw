#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

TEST_CASE("host_unique_ptr", "[cudaMemTools]") {
  exitSansCUDADevices();

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  SECTION("Single element") {
    auto ptr = cudautils::make_host_unique<int>(stream);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Reset") {
    auto ptr = cudautils::make_host_unique<int>(stream);
    REQUIRE(ptr != nullptr);

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr = cudautils::make_host_unique<int[]>(10, stream);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 27;  // 8**9
    auto ptr = cudautils::make_host_unique<char[]>(maxSize, stream);
    ptr.reset();
    REQUIRE_THROWS(ptr = cudautils::make_host_unique<char[]>(maxSize + 1, stream));
  }

  cudaCheck(cudaStreamDestroy(stream));
}
