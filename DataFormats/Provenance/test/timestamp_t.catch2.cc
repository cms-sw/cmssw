/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/Timestamp.h"

using namespace edm;

TEST_CASE("Timestamp", "[Timestamp]") {
  SECTION("constructTest") {
    const TimeValue_t t = 2;

    Timestamp temp(t);

    REQUIRE(temp.value() == t);

    REQUIRE(Timestamp::invalidTimestamp() < Timestamp::beginOfTime());
    REQUIRE(Timestamp::beginOfTime() < Timestamp::endOfTime());
  }

  SECTION("comparisonTest") {
    const Timestamp small(1);
    const Timestamp med(2);

    REQUIRE(small < med);
    REQUIRE(small <= med);
    REQUIRE(!(small == med));
    REQUIRE(small != med);
    REQUIRE(!(small > med));
    REQUIRE(!(small >= med));
  }
}
