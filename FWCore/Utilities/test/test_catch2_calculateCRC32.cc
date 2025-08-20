#include "catch.hpp"
#include "FWCore/Utilities/interface/calculateCRC32.h"

TEST_CASE("test cms::calculateCRC32", "[calculateCRC32]") {
  SECTION("known") {
    auto crc32 = cms::calculateCRC32("type_label_instance_process");

    // This known result was calculated using python as a cross check
    unsigned int knownResult = 1215348599;
    REQUIRE(crc32 == knownResult);
  }
  SECTION("empty") {
    auto emptyString_crc32 = cms::calculateCRC32("");
    REQUIRE(emptyString_crc32 == 0);
  }
}
