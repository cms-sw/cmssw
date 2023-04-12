// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"

TEST_CASE("HeterogeneousCore/ROCmUtilities testRequireROCmDevices", "[testRequireROCmDevices]") {
  SECTION("Test requireDevices()") {
    cms::rocmtest::requireDevices();

    int devices = 0;
    hipCheck(hipGetDeviceCount(&devices));

    REQUIRE(devices > 0);
  }
}
