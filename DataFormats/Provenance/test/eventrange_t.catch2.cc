/*
 *  eventrange_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Eric Vaandering December 2008.
 *
 */

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/EventID.h"

using namespace edm;

TEST_CASE("EventRange", "[EventRange]") {
  SECTION("constructTest") {
    RunNumber_t const rb = 1;
    RunNumber_t const re = 2;
    EventNumber_t const lb{30123456789ull};
    EventNumber_t const le{40123456789ull};

    EventRange normal(rb, 1, lb, re, 1, le);
    EventRange maxed(rb, 1, 0, re, 1, 0);

    REQUIRE(normal.startRun() == rb);
    REQUIRE(normal.endRun() == re);
    REQUIRE(normal.startEvent() == lb);
    REQUIRE(normal.endEvent() == le);
    REQUIRE(maxed.startEventID().event() == EventID::maxEventNumber());
    REQUIRE(maxed.endEventID().event() == EventID::maxEventNumber());
  }

  SECTION("comparisonTest") {
    EventID const small(1, 1, 1);
    EventID const med(7, 1, 2);
    EventID const large(8, 1, 10);
    EventID const larger(10, 1, 1);
    EventRange const normal(5, 1, 1, 8, 1, 1);
    EventRange const maxed(5, 1, 1, 8, 1, 0);

    REQUIRE(!contains(normal, small));
    REQUIRE(contains(normal, med));
    REQUIRE(!contains(normal, large));
    REQUIRE(!contains(normal, larger));

    REQUIRE(!contains(maxed, small));
    REQUIRE(contains(maxed, med));
    REQUIRE(contains(maxed, large));
    REQUIRE(!contains(maxed, larger));
  }

  SECTION("overlapTest") {
    EventRange const normal(5, 1, 1, 8, 1, 1);
    EventRange const small(6, 1, 1, 7, 1, 1);
    EventRange const large(3, 1, 1, 10, 1, 1);
    EventRange const early(3, 1, 1, 6, 1, 1);
    EventRange const late(7, 1, 1, 10, 1, 1);

    REQUIRE(contains(normal, normal));
    REQUIRE(contains(normal, small));
    REQUIRE(!contains(normal, large));
    REQUIRE(!contains(normal, early));
    REQUIRE(!contains(normal, late));
    REQUIRE(!contains(small, normal));
    REQUIRE(contains(small, small));
    REQUIRE(!contains(small, large));
    REQUIRE(!contains(small, early));
    REQUIRE(!contains(small, late));
    REQUIRE(contains(large, normal));
    REQUIRE(contains(large, small));
    REQUIRE(contains(large, large));
    REQUIRE(contains(large, early));
    REQUIRE(contains(large, late));
    REQUIRE(!contains(early, normal));
    REQUIRE(!contains(early, small));
    REQUIRE(!contains(early, large));
    REQUIRE(contains(early, early));
    REQUIRE(!contains(early, late));
    REQUIRE(!contains(late, normal));
    REQUIRE(!contains(late, small));
    REQUIRE(!contains(late, large));
    REQUIRE(!contains(late, early));
    REQUIRE(contains(late, late));
    REQUIRE(overlaps(normal, normal));
    REQUIRE(overlaps(normal, small));
    REQUIRE(overlaps(normal, large));
    REQUIRE(overlaps(normal, early));
    REQUIRE(overlaps(normal, late));
    REQUIRE(overlaps(small, normal));
    REQUIRE(overlaps(small, small));
    REQUIRE(overlaps(small, large));
    REQUIRE(overlaps(small, early));
    REQUIRE(overlaps(small, late));
    REQUIRE(overlaps(large, normal));
    REQUIRE(overlaps(large, small));
    REQUIRE(overlaps(large, large));
    REQUIRE(overlaps(large, early));
    REQUIRE(overlaps(large, late));
    REQUIRE(overlaps(early, normal));
    REQUIRE(overlaps(early, small));
    REQUIRE(overlaps(early, large));
    REQUIRE(overlaps(early, early));
    REQUIRE(!overlaps(early, late));
    REQUIRE(overlaps(late, normal));
    REQUIRE(overlaps(late, small));
    REQUIRE(overlaps(late, large));
    REQUIRE(!overlaps(late, early));
    REQUIRE(overlaps(late, late));
    REQUIRE(distinct(early, late));
    REQUIRE(distinct(late, early));
  }
}
