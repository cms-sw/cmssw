#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"

#include <sstream>
#include <string>
#include <vector>

TEST_CASE("StoredProcessBlockHelper", "[StoredProcessBlockHelper]") {
  SECTION("Default construction") {
    edm::StoredProcessBlockHelper storedProcessBlockHelper;
    REQUIRE(storedProcessBlockHelper.processesWithProcessBlockProducts().empty());
    REQUIRE(storedProcessBlockHelper.processBlockCacheIndices().empty());

    edm::StoredProcessBlockHelper const* storedProcessBlockHelperConstPtr = &storedProcessBlockHelper;
    REQUIRE(storedProcessBlockHelperConstPtr->processesWithProcessBlockProducts().empty());
    REQUIRE(storedProcessBlockHelperConstPtr->processBlockCacheIndices().empty());

    edm::EventToProcessBlockIndexes eventToProcessBlockIndexes;
    REQUIRE(eventToProcessBlockIndexes.index() == 0);
    eventToProcessBlockIndexes.setIndex(2);
    REQUIRE(eventToProcessBlockIndexes.index() == 2);
  }

  SECTION("Constructor") {
    std::vector<std::string> testStrings{"test1", "test2", "test3"};

    edm::StoredProcessBlockHelper storedProcessBlockHelper(testStrings);
    REQUIRE(storedProcessBlockHelper.processesWithProcessBlockProducts() == testStrings);
    REQUIRE(storedProcessBlockHelper.processBlockCacheIndices().empty());

    edm::StoredProcessBlockHelper const* storedProcessBlockHelperConstPtr = &storedProcessBlockHelper;
    REQUIRE(storedProcessBlockHelperConstPtr->processesWithProcessBlockProducts() == testStrings);
    REQUIRE(storedProcessBlockHelperConstPtr->processBlockCacheIndices().empty());

    std::vector<unsigned int> testIndices{1, 10, 100};
    storedProcessBlockHelper.processBlockCacheIndices() = testIndices;
    REQUIRE(storedProcessBlockHelper.processBlockCacheIndices() == testIndices);
    REQUIRE(storedProcessBlockHelperConstPtr->processBlockCacheIndices() == testIndices);
  }
}
