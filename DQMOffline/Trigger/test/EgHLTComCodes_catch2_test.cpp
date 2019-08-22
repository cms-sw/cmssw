#include "catch.hpp"

#include "DQMOffline/Trigger/interface/EgHLTComCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"

TEST_CASE("EgHLTComCodes", "[EgHLTComCodes]") {
  egHLT::ComCodes codes;

  constexpr unsigned int kFoo = 0b1;
  constexpr unsigned int kBar = 0b10;
  constexpr unsigned int kIsh = 0b100;
  constexpr unsigned int kTar = 0b1000;
  constexpr unsigned int kash = 0b10000;

  codes.setCode("Foo", kFoo);
  codes.setCode("Bar", kBar);
  codes.setCode("Ish", kIsh);
  codes.setCode("Tar", kTar);
  codes.setCode("ash", kash);
  codes.sort();

  SECTION("Sorted") {
    REQUIRE(codes.getCode("ash") == kash);
    REQUIRE(codes.getCode("Bar") == kBar);
    REQUIRE(codes.getCode("Foo") == kFoo);
    REQUIRE(codes.getCode("Ish") == kIsh);
    REQUIRE(codes.getCode("Tar") == kTar);
  }

  SECTION("Select multiple") {
    REQUIRE(codes.getCode("ash:Ish") == (kash | kIsh));
    REQUIRE(codes.getCode("Bar:Foo:Tar") == (kBar | kFoo | kTar));
    REQUIRE(codes.getCode("Tar:Foo:Bar") == (kTar | kFoo | kBar));
  }

  SECTION("Missing") {
    REQUIRE(codes.getCode("BAD") == 0);
    REQUIRE(codes.getCode("Tar:BAD:Bar") == (kTar | kBar));

    //no partial match
    REQUIRE(codes.getCode("as") == 0);
    REQUIRE(codes.getCode("ashton") == 0);
  }
}

TEST_CASE("EgHLTCutCodes", "[EgHLTCutCodes]") {
  SECTION("get good codes") {
    REQUIRE(egHLT::EgCutCodes::getCode("et") == egHLT::EgCutCodes::ET);
    REQUIRE(egHLT::EgCutCodes::getCode("maxr9") == egHLT::EgCutCodes::MAXR9);
  }
}
