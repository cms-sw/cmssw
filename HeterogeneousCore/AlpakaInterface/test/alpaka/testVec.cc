#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestVec) "]";

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestVec), s_tag) {
  SECTION("Vec1D") {
    alpaka_common::Vec1D s1 = 100;
    alpaka_common::Vec1D s2 = 42;
    alpaka_common::Vec1D si = 42;
    REQUIRE(alpaka::elementwise_min(s1, s2) == si);
  }

  SECTION("Vec3D") {
    alpaka_common::Vec3D v1{100, 10, 10};
    alpaka_common::Vec3D v2{42, 42, 1};
    alpaka_common::Vec3D vi{42, 10, 1};
    REQUIRE(alpaka::elementwise_min(v1, v2) == vi);
  }
}
