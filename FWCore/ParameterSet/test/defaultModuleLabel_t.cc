#include "catch2/catch_all.hpp"

#include "FWCore/ParameterSet/interface/defaultModuleLabel.h"

TEST_CASE("DefaultModuleLabel", "[ParameterSet]") {
  SECTION("test") {
    REQUIRE(edm::defaultModuleLabel("Dummy") == "dummy");
    REQUIRE(edm::defaultModuleLabel("DummyCamelCaps") == "dummyCamelCaps");
    REQUIRE(edm::defaultModuleLabel("ALLCAPS") == "allcaps");
    REQUIRE(edm::defaultModuleLabel("STARTCaps") == "startCaps");
    REQUIRE(edm::defaultModuleLabel("colons::Test") == "colonsTest");
  }
}
