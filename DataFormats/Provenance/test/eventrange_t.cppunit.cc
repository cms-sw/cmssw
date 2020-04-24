/*
 *  eventrange_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Eric Vaandering December 2008.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/EventID.h"

using namespace edm;

class testEventRange: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(testEventRange);

   CPPUNIT_TEST(constructTest);
   CPPUNIT_TEST(comparisonTest);
   CPPUNIT_TEST(overlapTest);
//   CPPUNIT_TEST(iterationTest);

   CPPUNIT_TEST_SUITE_END();
public:
   void setUp(){}
   void tearDown(){}

   void constructTest();
   void comparisonTest();
   void overlapTest();
//   void iterationTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventRange);


void testEventRange::constructTest() {
   RunNumber_t const rb = 1;
   RunNumber_t const re = 2;
   EventNumber_t const lb { 30123456789ull };
   EventNumber_t const le { 40123456789ull };

   EventRange normal(rb, 1, lb, re, 1, le);
   EventRange maxed(rb, 1, 0, re, 1, 0);

   CPPUNIT_ASSERT(normal.startRun() == rb);
   CPPUNIT_ASSERT(normal.endRun()   == re);
   CPPUNIT_ASSERT(normal.startEvent() == lb);
   CPPUNIT_ASSERT(normal.endEvent()   == le);
   CPPUNIT_ASSERT(maxed.startEventID().event() == EventID::maxEventNumber());
   CPPUNIT_ASSERT(maxed.endEventID().event() == EventID::maxEventNumber());
}

void testEventRange::comparisonTest() {
    EventID const small(1, 1, 1);
    EventID const med(7, 1, 2);
    EventID const large(8, 1, 10);
    EventID const larger(10,1, 1);
    EventRange const normal(5,1,1,8,1,1);
    EventRange const maxed(5,1,1,8,1,0);

    CPPUNIT_ASSERT(!contains(normal,small));
    CPPUNIT_ASSERT(contains(normal,med));
    CPPUNIT_ASSERT(!contains(normal,large));
    CPPUNIT_ASSERT(!contains(normal,larger));

    CPPUNIT_ASSERT(!contains(maxed,small));
    CPPUNIT_ASSERT(contains(maxed,med));
    CPPUNIT_ASSERT(contains(maxed,large));
    CPPUNIT_ASSERT(!contains(maxed,larger));
}

void testEventRange::overlapTest() {

    EventRange const normal(5,1,1,8,1,1);
    EventRange const small(6,1,1,7,1,1);
    EventRange const large(3,1,1,10,1,1);
    EventRange const early(3,1,1,6,1,1);
    EventRange const late(7,1,1,10,1,1);

    CPPUNIT_ASSERT(contains(normal,normal));
    CPPUNIT_ASSERT(contains(normal,small));
    CPPUNIT_ASSERT(!contains(normal,large));
    CPPUNIT_ASSERT(!contains(normal,early));
    CPPUNIT_ASSERT(!contains(normal,late));
    CPPUNIT_ASSERT(!contains(small,normal));
    CPPUNIT_ASSERT(contains(small,small));
    CPPUNIT_ASSERT(!contains(small,large));
    CPPUNIT_ASSERT(!contains(small,early));
    CPPUNIT_ASSERT(!contains(small,late));
    CPPUNIT_ASSERT(contains(large,normal));
    CPPUNIT_ASSERT(contains(large,small));
    CPPUNIT_ASSERT(contains(large,large));
    CPPUNIT_ASSERT(contains(large,early));
    CPPUNIT_ASSERT(contains(large,late));
    CPPUNIT_ASSERT(!contains(early,normal));
    CPPUNIT_ASSERT(!contains(early,small));
    CPPUNIT_ASSERT(!contains(early,large));
    CPPUNIT_ASSERT(contains(early,early));
    CPPUNIT_ASSERT(!contains(early,late));
    CPPUNIT_ASSERT(!contains(late,normal));
    CPPUNIT_ASSERT(!contains(late,small));
    CPPUNIT_ASSERT(!contains(late,large));
    CPPUNIT_ASSERT(!contains(late,early));
    CPPUNIT_ASSERT(contains(late,late));
    CPPUNIT_ASSERT(overlaps(normal,normal));
    CPPUNIT_ASSERT(overlaps(normal,small));
    CPPUNIT_ASSERT(overlaps(normal,large));
    CPPUNIT_ASSERT(overlaps(normal,early));
    CPPUNIT_ASSERT(overlaps(normal,late));
    CPPUNIT_ASSERT(overlaps(small,normal));
    CPPUNIT_ASSERT(overlaps(small,small));
    CPPUNIT_ASSERT(overlaps(small,large));
    CPPUNIT_ASSERT(overlaps(small,early));
    CPPUNIT_ASSERT(overlaps(small,late));
    CPPUNIT_ASSERT(overlaps(large,normal));
    CPPUNIT_ASSERT(overlaps(large,small));
    CPPUNIT_ASSERT(overlaps(large,large));
    CPPUNIT_ASSERT(overlaps(large,early));
    CPPUNIT_ASSERT(overlaps(large,late));
    CPPUNIT_ASSERT(overlaps(early,normal));
    CPPUNIT_ASSERT(overlaps(early,small));
    CPPUNIT_ASSERT(overlaps(early,large));
    CPPUNIT_ASSERT(overlaps(early,early));
    CPPUNIT_ASSERT(!overlaps(early,late));
    CPPUNIT_ASSERT(overlaps(late,normal));
    CPPUNIT_ASSERT(overlaps(late,small));
    CPPUNIT_ASSERT(overlaps(late,large));
    CPPUNIT_ASSERT(!overlaps(late,early));
    CPPUNIT_ASSERT(overlaps(late,late));
    CPPUNIT_ASSERT(distinct(early,late));
    CPPUNIT_ASSERT(distinct(late,early));
}
