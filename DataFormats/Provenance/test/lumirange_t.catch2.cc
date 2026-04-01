/*
 *  lumirange_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Eric Vaandering December 2008.
 *
 */

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

using namespace edm;

TEST_CASE("LumiRange", "[LumiRange]") {
  SECTION("constructTest") {
    const RunNumber_t rb = 1;
    const RunNumber_t re = 2;
    const LuminosityBlockNumber_t lb = 3;
    const LuminosityBlockNumber_t le = 4;

    LuminosityBlockRange normal(rb, lb, re, le);
    LuminosityBlockRange maxed(rb, 0, re, 0);
    LuminosityBlockID dummy;

    REQUIRE(normal.startRun() == rb);
    REQUIRE(normal.endRun() == re);
    REQUIRE(normal.startLumi() == lb);
    REQUIRE(normal.endLumi() == le);
    REQUIRE(maxed.startLumiID().luminosityBlock() == dummy.maxLuminosityBlockNumber());
    REQUIRE(maxed.endLumiID().luminosityBlock() == dummy.maxLuminosityBlockNumber());
  }

  SECTION("comparisonTest") {
    const LuminosityBlockID small(1, 1);
    const LuminosityBlockID med(7, 2);
    const LuminosityBlockID large(8, 10);
    const LuminosityBlockID larger(10, 1);
    const LuminosityBlockRange normal(5, 1, 8, 1);
    const LuminosityBlockRange maxed(5, 1, 8, 0);

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
    const LuminosityBlockRange normal(5, 1, 8, 1);
    const LuminosityBlockRange small(6, 1, 7, 1);
    const LuminosityBlockRange large(3, 1, 10, 1);
    const LuminosityBlockRange early(3, 1, 6, 1);
    const LuminosityBlockRange late(7, 1, 10, 1);

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
  }
}

// void testEventID::iterationTest() {
//    EventID first = EventID::firstValidEvent();
//
//    EventID second = first.next();
//    REQUIRE(first < second);
//    REQUIRE(first == (second.previous()));
//
//    EventID run2(2, 0);
//    REQUIRE(run2 < run2.nextRun());
//    REQUIRE(run2 > run2.previousRunLastEvent());
//    REQUIRE(first < run2.previousRunLastEvent());
//    REQUIRE(run2 < first.nextRunFirstEvent());
//
//    EventID run2Last(2, EventID::maxEventNumber());
//    REQUIRE(run2Last.next() == run2Last.nextRunFirstEvent());
// }
