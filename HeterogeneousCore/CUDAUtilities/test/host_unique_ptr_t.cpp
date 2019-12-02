#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

TEST_CASE("host_unique_ptr", "[cudaMemTools]") {
  if (not cms::cudatest::testDevices()) {
    return;
  }

  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  SECTION("Single element") {
    auto ptr = cms::cuda::make_host_unique<int>(stream);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Reset") {
    auto ptr = cms::cuda::make_host_unique<int>(stream);
    REQUIRE(ptr != nullptr);
    cudaCheck(cudaStreamSynchronize(stream));

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr = cms::cuda::make_host_unique<int[]>(10, stream);
    REQUIRE(ptr != nullptr);
    cudaCheck(cudaStreamSynchronize(stream));

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 30;  // 8**10
    auto ptr = cms::cuda::make_host_unique<char[]>(maxSize, stream);
    ptr.reset();
    REQUIRE_THROWS(ptr = cms::cuda::make_host_unique<char[]>(maxSize + 1, stream));
  }

  cudaCheck(cudaStreamDestroy(stream));
}
