#include "catch.hpp"
#include "FWCore/Utilities/interface/CRC32Calculator.h"

TEST_CASE("test cms::CRC32Calculator", "[CRC32Calculator]") {
  SECTION("known") {
    cms::CRC32Calculator crc32("type_label_instance_process");

    // This known result was calculated using python as a cross check
    unsigned int knownResult = 1215348599;
    REQUIRE(crc32.checksum() == knownResult);
  }
  SECTION("empty") {
    cms::CRC32Calculator emptyString_crc32("");
    REQUIRE(emptyString_crc32.checksum() == 0);
  }
}
