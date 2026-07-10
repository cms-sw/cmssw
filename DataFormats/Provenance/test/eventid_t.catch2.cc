/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/EventID.h"

using namespace edm;

TEST_CASE("EventID", "[EventID]") {
  SECTION("constructTest") {
    EventID eventID;
    REQUIRE(eventID.run() == 0U);
    REQUIRE(eventID.luminosityBlock() == 0U);
    REQUIRE(eventID.event() == 0U);

    const RunNumber_t rt = 3;
    const LuminosityBlockNumber_t lt = 2;
    const EventNumber_t et = 10123456789;

    EventID temp(rt, lt, et);

    REQUIRE(temp.run() == rt);
    REQUIRE(temp.luminosityBlock() == lt);
    REQUIRE(temp.event() == et);

    REQUIRE(EventID::maxRunNumber() == 0xFFFFFFFF);
    REQUIRE(EventID::maxLuminosityBlockNumber() == 0xFFFFFFFF);
    REQUIRE(EventID::maxEventNumber() == 0xFFFFFFFFFFFFFFFF);
  }

  SECTION("comparisonTest") {
    const EventID small(1, 4, 1);
    const EventID med(2, 3, 2);
    const EventID med2(2, 3, 2);
    const EventID large(3, 1, 3);
    const EventID larger(3, 2, 1);
    const EventID largest(3, 2, 2);

    REQUIRE(small < med);
    REQUIRE(!(med < small));
    REQUIRE(small <= med);
    REQUIRE(!(small == med));
    REQUIRE(small != med);
    REQUIRE(!(small > med));
    REQUIRE(!(small >= med));

    REQUIRE(!(med <= small));
    REQUIRE(!(med == small));
    REQUIRE(med != small);
    REQUIRE(med > small);
    REQUIRE(med >= small);

    REQUIRE(med2 == med);
    REQUIRE(med2 <= med);
    REQUIRE(med2 >= med);
    REQUIRE(!(med2 != med));
    REQUIRE(!(med2 < med));
    REQUIRE(!(med2 > med));

    REQUIRE(med < large);
    REQUIRE(med <= large);
    REQUIRE(!(med == large));
    REQUIRE(med != large);
    REQUIRE(!(med > large));
    REQUIRE(!(med >= large));

    REQUIRE(large < largest);
    REQUIRE(!(largest < large));
    REQUIRE(large <= largest);
    REQUIRE(!(large == largest));
    REQUIRE(large != largest);
    REQUIRE(!(large > largest));
    REQUIRE(!(large >= largest));

    REQUIRE(larger < largest);
    REQUIRE(!(largest < larger));
  }

  SECTION("iterationTest") {
    EventID first = EventID::firstValidEvent();

    EventID second = first.next(1);
    REQUIRE(first < second);
    REQUIRE(first == (second.previous(1)));

    EventID run2(2, 1, 0);
    REQUIRE(run2 < run2.nextRun(1));
    REQUIRE(run2 > run2.previousRunLastEvent(1));
    REQUIRE(first < run2.previousRunLastEvent(1));
    REQUIRE(run2 < first.nextRunFirstEvent(1));

    EventID run2Last(2, 1, EventID::maxEventNumber());
    REQUIRE(run2Last.next(1) == run2Last.nextRunFirstEvent(1));
  }
}
