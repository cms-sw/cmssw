#include "catch.hpp"
#include "CondCore/CondHDF5ESSource/plugins/IOVSyncValue.h"
//can't link to plugin so must include source
#include "CondCore/CondHDF5ESSource/plugins/IOVSyncValue.cc"

using namespace cond::hdf5;

TEST_CASE("test cond::hdf5::findMatchingFirst", "[findMatchingFirst]") {
  SECTION("empty IOVs") {
    std::vector<IOVSyncValue> iovs;

    auto itFind = findMatchingFirst(iovs, {1, 0});
    REQUIRE(itFind == iovs.end());
  }

  SECTION("First element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {1, 0});
    REQUIRE(itFind == iovs.begin());
  }

  SECTION("Second element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {10, 0});
    REQUIRE(itFind == iovs.begin() + 1);
  }

  SECTION("Last element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {20, 0});
    REQUIRE(itFind == iovs.begin() + 2);
  }

  SECTION("Between first and second element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {5, 0});
    REQUIRE(itFind == iovs.begin());
  }

  SECTION("Between second and third element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {15, 0});
    REQUIRE(itFind == iovs.begin() + 1);
  }

  SECTION("After last element") {
    std::vector<IOVSyncValue> iovs = {{1, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {25, 0});
    REQUIRE(itFind == iovs.begin() + 2);
  }

  SECTION("Before first element") {
    std::vector<IOVSyncValue> iovs = {{2, 0}, {10, 0}, {20, 0}};

    auto itFind = findMatchingFirst(iovs, {1, 0});
    REQUIRE(itFind == iovs.end());
  }
}
