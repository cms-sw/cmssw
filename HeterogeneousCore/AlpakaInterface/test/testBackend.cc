#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"

TEST_CASE("Test cms::alpakatools::toBackend", "cms::alpakatools::Backend") {
  SECTION("Valid string") {
    REQUIRE(cms::alpakatools::toBackend("SerialSync") == cms::alpakatools::Backend::SerialSync);
    REQUIRE(cms::alpakatools::toBackend("CudaAsync") == cms::alpakatools::Backend::CudaAsync);
    REQUIRE(cms::alpakatools::toBackend("ROCmAsync") == cms::alpakatools::Backend::ROCmAsync);
    REQUIRE(cms::alpakatools::toBackend("TbbAsync") == cms::alpakatools::Backend::TbbAsync);
  }
  SECTION("Invalid string") {
    REQUIRE_THROWS_WITH(cms::alpakatools::toBackend("Nonexistent"),
                        Catch::Contains("EnumNotFound") and Catch::Contains("Invalid backend name"));
  }
}

TEST_CASE("Test cms::alpakatools::toString", "cms::alpakatools::Backend") {
  SECTION("Valid enum") {
    REQUIRE(cms::alpakatools::toString(cms::alpakatools::Backend::SerialSync) == "SerialSync");
    REQUIRE(cms::alpakatools::toString(cms::alpakatools::Backend::CudaAsync) == "CudaAsync");
    REQUIRE(cms::alpakatools::toString(cms::alpakatools::Backend::ROCmAsync) == "ROCmAsync");
    REQUIRE(cms::alpakatools::toString(cms::alpakatools::Backend::TbbAsync) == "TbbAsync");
  }
  SECTION("Invalid enum") {
    REQUIRE_THROWS_WITH(cms::alpakatools::toString(cms::alpakatools::Backend::size),
                        Catch::Contains("InvalidEnumValue") and Catch::Contains("Invalid backend enum value"));
  }
}
