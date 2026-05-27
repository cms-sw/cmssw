#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ThreadHandoff.h"
#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

using namespace edm;
TEST_CASE("Test edm::ThreadHandoff", "[ThreadHandoff]") {
  constexpr unsigned stackSize = 1024 * 1024;  // MB

  SECTION("Do nothing") { ThreadHandoff th(stackSize); }
  SECTION("Simple") {
    ThreadHandoff th(stackSize);
    bool value = false;
    th.runAndWait([&value]() { value = true; });
    REQUIRE(value == true);
  }

  SECTION("Exception") {
    ThreadHandoff th(stackSize);
    REQUIRE_THROWS_AS(th.runAndWait([]() { throw cms::Exception("Test"); }), cms::Exception);
  }
}
