#include "catch.hpp"
#include "FWCore/Utilities/interface/Guid.h"

TEST_CASE(
    "test edm::Guid"
    "[Guid]") {
  edm::Guid guid;
  auto guidString = guid.toString();
  auto guidBinary = guid.toBinary();

  SECTION("copying from string") {
    edm::Guid guid2(guidString, false);
    REQUIRE(guid == guid2);
    REQUIRE(guidString == guid2.toString());
    REQUIRE(guidBinary == guid2.toBinary());
  }
  SECTION("copy constructor") {
    edm::Guid guid3(guid);
    REQUIRE(guid == guid3);
    REQUIRE(guidString == guid3.toString());
    REQUIRE(guidBinary == guid3.toBinary());
  }
  SECTION("copy from binary") {
    edm::Guid guid4(guidBinary, true);

    REQUIRE(guid == guid4);
    REQUIRE(guidString == guid4.toString());
    REQUIRE(guidBinary == guid4.toBinary());
  }
  SECTION("uniqueness") {
    edm::Guid otherGuid;
    REQUIRE(otherGuid != guid);
    edm::Guid otherBinaryGuid{otherGuid.toBinary(), true};
    REQUIRE(otherBinaryGuid == otherGuid);
    REQUIRE(otherBinaryGuid != guid);
  }
}
