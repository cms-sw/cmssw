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
//   CPPUNIT_TEST(iterationTest);

   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}

   void constructTest();
   void comparisonTest();
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

    CPPUNIT_ASSERT(!normal.contains(small));
    CPPUNIT_ASSERT(normal.contains(med));
    CPPUNIT_ASSERT(!normal.contains(large));
    CPPUNIT_ASSERT(!normal.contains(larger));

    CPPUNIT_ASSERT(!maxed.contains(small));
    CPPUNIT_ASSERT(maxed.contains(med));
    CPPUNIT_ASSERT(maxed.contains(large));
    CPPUNIT_ASSERT(!maxed.contains(larger));

}
