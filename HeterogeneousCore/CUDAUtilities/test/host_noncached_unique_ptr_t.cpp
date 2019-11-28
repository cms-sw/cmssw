#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

TEST_CASE("host_noncached_unique_ptr", "[cudaMemTools]") {
  exitSansCUDADevices();

  SECTION("Single element") {
    auto ptr1 = cudautils::make_host_noncached_unique<int>();
    REQUIRE(ptr1 != nullptr);
    auto ptr2 = cudautils::make_host_noncached_unique<int>(cudaHostAllocPortable | cudaHostAllocWriteCombined);
    REQUIRE(ptr2 != nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr1 = cudautils::make_host_noncached_unique<int[]>(10);
    REQUIRE(ptr1 != nullptr);
    auto ptr2 = cudautils::make_host_noncached_unique<int[]>(10, cudaHostAllocPortable | cudaHostAllocWriteCombined);
    REQUIRE(ptr2 != nullptr);
  }
}
