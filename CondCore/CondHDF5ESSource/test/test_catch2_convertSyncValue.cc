#include "catch.hpp"
#include "CondCore/CondHDF5ESSource/plugins/convertSyncValue.h"

using namespace cond::hdf5;

TEST_CASE("test cond::hdf5::convertSyncValue", "[convertSyncValue]") {
  SECTION("run_lumi") {
    edm::IOVSyncValue edmSync{edm::EventID{5, 8, 0}};
    auto condSync = convertSyncValue(edmSync, true);
    REQUIRE(condSync.high_ == 5);
    REQUIRE(condSync.low_ == 8);

    REQUIRE(edmSync == convertSyncValue(condSync, true));
  }
  SECTION("time") {
    edm::IOVSyncValue edmSync{edm::Timestamp{(static_cast<uint64_t>(5) << 32) + 8}};
    auto condSync = convertSyncValue(edmSync, false);
    REQUIRE(condSync.high_ == 5);
    REQUIRE(condSync.low_ == 8);

    REQUIRE(edmSync == convertSyncValue(condSync, false));
  }
}
