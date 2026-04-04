/*
 *  interval_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/30/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "catch2/catch_all.hpp"

using edm::IOVSyncValue;
using edm::Timestamp;
using edm::ValidityInterval;

TEST_CASE("Interval", "[Framework][EventSetup]") {
  SECTION("comparisonTest") {
    const IOVSyncValue invalid(IOVSyncValue::invalidIOVSyncValue());

    const Timestamp t_1(1);
    const IOVSyncValue one(t_1);
    const Timestamp t_2(2);
    const IOVSyncValue two(t_2);

    REQUIRE(invalid == IOVSyncValue::invalidIOVSyncValue());
    REQUIRE(one == IOVSyncValue(t_1));

    REQUIRE(invalid != one);

    REQUIRE(one < two);
    REQUIRE(!(one > two));
    REQUIRE(two > one);
    REQUIRE(!(two < one));

    REQUIRE(one != two);
    REQUIRE(!(one == two));

    REQUIRE(one <= two);
    REQUIRE(one <= one);
    REQUIRE(one >= one);
    REQUIRE(!(one >= two));
  }

  SECTION("timestampAssignmentTest") {
    const Timestamp t_1(1);
    const IOVSyncValue one(t_1);

    IOVSyncValue temp(IOVSyncValue::invalidIOVSyncValue());
    REQUIRE(temp != one);
    temp = one;
    REQUIRE(temp == one);
  }

  SECTION("intervalAssignmentTest") {
    ValidityInterval temp;
    const Timestamp t_1(1);
    const IOVSyncValue s_1(t_1);
    const ValidityInterval oneAndTwo(s_1, IOVSyncValue(Timestamp(2)));

    REQUIRE(temp != oneAndTwo);
    REQUIRE(!(temp == oneAndTwo));

    temp = oneAndTwo;
    REQUIRE(temp == oneAndTwo);
  }
}
