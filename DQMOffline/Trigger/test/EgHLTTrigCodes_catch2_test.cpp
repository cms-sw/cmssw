#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

TEST_CASE("EgHLTTrigCodes", "[EgHLTTrigCodes]") {
  std::vector<std::string> names = {{"Foo"}, {"Bar"}, {"Ish"}, {"Tar"}, {"ash"}};
  constexpr unsigned int kFoo = 0b1;
  constexpr unsigned int kBar = 0b10;
  constexpr unsigned int kIsh = 0b100;
  constexpr unsigned int kTar = 0b1000;
  constexpr unsigned int kash = 0b10000;

  using bits = egHLT::TrigCodes::TrigBitSet;
  SECTION("Sorted") {
    //This will sort to
    // ash, Bar, Foo Ish Tar

    std::unique_ptr<egHLT::TrigCodes> codes(egHLT::TrigCodes::makeCodes(names));

    REQUIRE(codes->getCode("ash") == bits(kash));
    REQUIRE(codes->getCode("Bar") == bits(kBar));
    REQUIRE(codes->getCode("Foo") == bits(kFoo));
    REQUIRE(codes->getCode("Ish") == bits(kIsh));
    REQUIRE(codes->getCode("Tar") == bits(kTar));
  }

  SECTION("Select multiple") {
    std::unique_ptr<egHLT::TrigCodes> codes(egHLT::TrigCodes::makeCodes(names));
    REQUIRE(codes->getCode("ash:Ish") == bits(kash | kIsh));
    REQUIRE(codes->getCode("Bar:Foo:Tar") == bits(kBar | kFoo | kTar));
    REQUIRE(codes->getCode("Tar:Foo:Bar") == bits(kTar | kFoo | kBar));
  }

  SECTION("Missing") {
    std::unique_ptr<egHLT::TrigCodes> codes(egHLT::TrigCodes::makeCodes(names));
    REQUIRE(codes->getCode("BAD") == bits());
    REQUIRE(codes->getCode("Tar:BAD:Bar") == bits(kTar | kBar));

    //no partial match
    REQUIRE(codes->getCode("as") == bits());
    REQUIRE(codes->getCode("ashton") == bits());
  }
}
