/*
 *  eventid_t.catch2.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "catch2/catch_all.hpp"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;

TEST_CASE("IOVSyncValue", "[Framework][EventSetup]") {
  SECTION("constructTest") {
    {
      const EventID t(2, 0, 0);

      IOVSyncValue temp(t);

      REQUIRE(temp.eventID() == t);

      REQUIRE(IOVSyncValue::invalidIOVSyncValue() != temp);
      REQUIRE(!(IOVSyncValue::invalidIOVSyncValue() == temp));
      REQUIRE(IOVSyncValue::beginOfTime() < temp);
      REQUIRE(IOVSyncValue::endOfTime() > temp);
    }
    REQUIRE(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
    REQUIRE(IOVSyncValue::beginOfTime() < IOVSyncValue::endOfTime());

    {
      const EventID t(2, 3, 1);

      IOVSyncValue temp(t);

      REQUIRE(temp.eventID() == t);
      REQUIRE(temp.luminosityBlockNumber() == 3);

      REQUIRE(IOVSyncValue::invalidIOVSyncValue() != temp);
      REQUIRE(!(IOVSyncValue::invalidIOVSyncValue() == temp));
      REQUIRE(IOVSyncValue::beginOfTime() < temp);
      REQUIRE(IOVSyncValue::endOfTime() > temp);
    }
  }

  SECTION("constructTimeTest") {
    const Timestamp t(2);

    IOVSyncValue temp(t);

    REQUIRE(temp.time() == t);

    REQUIRE(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
    REQUIRE(IOVSyncValue::beginOfTime() < IOVSyncValue::endOfTime());

    REQUIRE(IOVSyncValue::invalidIOVSyncValue() != temp);
    REQUIRE(!(IOVSyncValue::invalidIOVSyncValue() == temp));
    REQUIRE(IOVSyncValue::beginOfTime() < temp);
    REQUIRE(IOVSyncValue::endOfTime() > temp);
  }

  SECTION("comparisonTest") {
    {
      const IOVSyncValue small(EventID(1, 1, 1));
      const IOVSyncValue med(EventID(2, 1, 2));

      REQUIRE(small.comparable(med));
      REQUIRE(small < med);
      REQUIRE(small <= med);
      REQUIRE(!(small == med));
      REQUIRE(small != med);
      REQUIRE(!(small > med));
      REQUIRE(!(small >= med));
    }
    {
      const IOVSyncValue small(EventID(2, 1, 1));
      const IOVSyncValue med(EventID(2, 1, 2));

      REQUIRE(small < med);
      REQUIRE(small <= med);
      REQUIRE(!(small == med));
      REQUIRE(small != med);
      REQUIRE(!(small > med));
      REQUIRE(!(small >= med));
    }
    {
      const IOVSyncValue small(EventID(2, 1, 2));
      const IOVSyncValue med(EventID(3, 1, 1));

      REQUIRE(small < med);
      REQUIRE(small <= med);
      REQUIRE(!(small == med));
      REQUIRE(small != med);
      REQUIRE(!(small > med));
      REQUIRE(!(small >= med));
    }
    {
      const IOVSyncValue small(EventID(2, 2, 1));
      const IOVSyncValue med(EventID(2, 2, 2));

      REQUIRE(small < med);
      REQUIRE(small <= med);
      REQUIRE(!(small == med));
      REQUIRE(small != med);
      REQUIRE(!(small > med));
      REQUIRE(!(small >= med));
    }
    {
      const IOVSyncValue small(EventID(2, 1, 3));
      const IOVSyncValue med(EventID(2, 2, 2));

      REQUIRE(small < med);
      REQUIRE(small <= med);
      REQUIRE(!(small == med));
      REQUIRE(small != med);
      REQUIRE(!(small > med));
      REQUIRE(!(small >= med));
    }
  }

  SECTION("comparisonTimeTest") {
    const IOVSyncValue small(Timestamp(1));
    const IOVSyncValue med(Timestamp(2));

    REQUIRE(small.comparable(med));
    REQUIRE(small < med);
    REQUIRE(small <= med);
    REQUIRE(!(small == med));
    REQUIRE(small != med);
    REQUIRE(!(small > med));
    REQUIRE(!(small >= med));
  }

  SECTION("invalidComparisonTest") {
    const IOVSyncValue timeBased(Timestamp(1));
    const IOVSyncValue eventBased(EventID(3, 2, 1));

    REQUIRE(!timeBased.comparable(eventBased));
    REQUIRE(!eventBased.comparable(timeBased));
    REQUIRE_THROWS_AS([&]() { return timeBased < eventBased; }(), cms::Exception);
    REQUIRE_THROWS_AS([&]() { return timeBased <= eventBased; }(), cms::Exception);
    REQUIRE(!(timeBased == eventBased));
    REQUIRE(timeBased != eventBased);
    REQUIRE_THROWS_AS([&]() { return timeBased > eventBased; }(), cms::Exception);
    REQUIRE_THROWS_AS([&]() { return timeBased >= eventBased; }(), cms::Exception);
  }
}
