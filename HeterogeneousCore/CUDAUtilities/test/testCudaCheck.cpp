// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

TEST_CASE("HeterogeneousCore/CUDAUtilities testCudaCheck", "[testCudaCheck]") {
  SECTION("Test CUDA_CHECK() driver API") {
    REQUIRE_NOTHROW(CUDA_CHECK(CUDA_SUCCESS));
    REQUIRE_THROWS(CUDA_CHECK(CUDA_ERROR_UNKNOWN));
  }
  SECTION("Test CUDA_CHECK() runtime API") {
    REQUIRE_NOTHROW(CUDA_CHECK(cudaSuccess));
    REQUIRE_THROWS(CUDA_CHECK(cudaErrorUnknown));
  }
}
