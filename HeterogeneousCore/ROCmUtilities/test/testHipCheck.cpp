// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

TEST_CASE("HeterogeneousCore/ROCmUtilities testHipCheck", "[testHipCheck]") {
  SECTION("Test hipCheck() API") {
    REQUIRE_NOTHROW(hipCheck(hipSuccess));
    REQUIRE_THROWS(hipCheck(hipErrorUnknown));
  }
}
