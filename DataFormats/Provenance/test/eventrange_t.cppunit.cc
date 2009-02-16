/*
 *  eventrange_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Eric Vaandering December 2008.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Provenance/interface/EventRange.h"

using namespace edm;

class testEventRange: public CppUnit::TestFixture
{
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


void testEventRange::constructTest()
{
   const RunNumber_t rb = 1;
   const RunNumber_t re = 2;
   const EventNumber_t lb = 3;
   const EventNumber_t le = 4;

   EventRange normal(rb, lb, re, le);
   EventRange maxed(rb, 0, re, 0);
   EventID    dummy;

   CPPUNIT_ASSERT(normal.startRun() == rb);
   CPPUNIT_ASSERT(normal.endRun()   == re);
   CPPUNIT_ASSERT(normal.startEvent() == lb);
   CPPUNIT_ASSERT(normal.endEvent()   == le);
   CPPUNIT_ASSERT(maxed.startEventID().event() == dummy.maxEventNumber());
   CPPUNIT_ASSERT(maxed.endEventID().event() == dummy.maxEventNumber());

}

void testEventRange::comparisonTest()
{
    const EventID small(1,1);
    const EventID med(7, 2);
    const EventID large(8,10);
    const EventID larger(10,1);
    const EventRange normal(5,1,8,1);
    const EventRange maxed(5,1,8,0);

    CPPUNIT_ASSERT(!contains(normal,small));
    CPPUNIT_ASSERT(contains(normal,med));
    CPPUNIT_ASSERT(!contains(normal,large));
    CPPUNIT_ASSERT(!contains(normal,larger));

    CPPUNIT_ASSERT(!contains(maxed,small));
    CPPUNIT_ASSERT(contains(maxed,med));
    CPPUNIT_ASSERT(contains(maxed,large));
    CPPUNIT_ASSERT(!contains(maxed,larger));

}

void testEventRange::overlapTest()
{

    const EventRange normal(5,1,8,1);
    const EventRange small(6,1,7,1);
    const EventRange large(3,1,10,1);
    const EventRange early(3,1,6,1);
    const EventRange late(7,1,10,1);

    CPPUNIT_ASSERT(contains(normal,small));
    CPPUNIT_ASSERT(!contains(normal,late));
    CPPUNIT_ASSERT(!contains(normal,early));
    CPPUNIT_ASSERT(!contains(normal,large));
    CPPUNIT_ASSERT(distinct(early,late));
    CPPUNIT_ASSERT(overlaps(normal,late));
    CPPUNIT_ASSERT(overlaps(normal,early));
    CPPUNIT_ASSERT(contains(large,early));
    CPPUNIT_ASSERT(contains(large,late));
    CPPUNIT_ASSERT(contains(large,large));

}
