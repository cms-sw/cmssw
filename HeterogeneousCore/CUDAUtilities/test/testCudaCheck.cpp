// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

TEST_CASE("HeterogeneousCore/CUDAUtilities testCudaCheck", "[testCudaCheck]") {
  SECTION("Test cudaCheck() driver API") {
    REQUIRE_NOTHROW(cudaCheck(CUDA_SUCCESS));
    REQUIRE_THROWS(cudaCheck(CUDA_ERROR_UNKNOWN));
  }
  SECTION("Test cudaCheck() runtime API") {
    REQUIRE_NOTHROW(cudaCheck(cudaSuccess));
    REQUIRE_THROWS(cudaCheck(cudaErrorUnknown));
  }
}
