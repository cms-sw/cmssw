#include "FWCore/Utilities/interface/path_configuration.h"
#include "catch.hpp"

using namespace edm::path_configuration;
TEST_CASE("Test path_configuration", "[path_configuration]") {
  SECTION("removeSchedulingTokensFromModuleLabel") {
    SECTION("no tokens") { REQUIRE("foo" == removeSchedulingTokensFromModuleLabel("foo")); }
    SECTION("+") { REQUIRE("foo" == removeSchedulingTokensFromModuleLabel("+foo")); }
    SECTION("-") { REQUIRE("foo" == removeSchedulingTokensFromModuleLabel("-foo")); }
    SECTION("|") { REQUIRE("foo" == removeSchedulingTokensFromModuleLabel("|foo")); }
    SECTION("!") { REQUIRE("foo" == removeSchedulingTokensFromModuleLabel("!foo")); }
  }
}
