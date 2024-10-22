// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

TEST_CASE("HeterogeneousCore/CUDAUtilities testRequireCUDADevices", "[testRequireCUDADevices]") {
  SECTION("Test requireDevices()") {
    cms::cudatest::requireDevices();

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));

    REQUIRE(devices > 0);
  }
}
