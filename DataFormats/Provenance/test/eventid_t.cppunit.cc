/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Provenance/interface/EventID.h"

using namespace edm;

class testEventID: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testEventID);
   
   CPPUNIT_TEST(constructTest);
   CPPUNIT_TEST(comparisonTest);
   CPPUNIT_TEST(iterationTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
   void setUp(){}
   void tearDown(){}
   
   void constructTest();
   void comparisonTest();
   void iterationTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventID);


void testEventID::constructTest()
 {
   EventID eventID;
   CPPUNIT_ASSERT(eventID.run() == 0U);
   CPPUNIT_ASSERT(eventID.luminosityBlock() == 0U);
   CPPUNIT_ASSERT(eventID.event() == 0U);

   const RunNumber_t rt = 3;
   const LuminosityBlockNumber_t lt = 2;
   const EventNumber_t et = 10123456789;

   EventID temp(rt, lt, et);
   
   CPPUNIT_ASSERT(temp.run() == rt);
   CPPUNIT_ASSERT(temp.luminosityBlock() == lt);
   CPPUNIT_ASSERT(temp.event() == et);

   CPPUNIT_ASSERT(EventID::maxRunNumber() == 0xFFFFFFFF);
   CPPUNIT_ASSERT(EventID::maxLuminosityBlockNumber() == 0xFFFFFFFF);
   CPPUNIT_ASSERT(EventID::maxEventNumber() == 0xFFFFFFFFFFFFFFFF);
}

void testEventID::comparisonTest()
{
   const EventID small(1, 4, 1);
   const EventID med(2, 3, 2);
   const EventID med2(2, 3, 2);
   const EventID large(3, 1, 3);
   const EventID larger(3, 2, 1);
   const EventID largest(3, 2, 2);
   
   CPPUNIT_ASSERT(small < med);
   CPPUNIT_ASSERT(!(med < small));
   CPPUNIT_ASSERT(small <= med);
   CPPUNIT_ASSERT(!(small == med));
   CPPUNIT_ASSERT(small != med);
   CPPUNIT_ASSERT(!(small > med));
   CPPUNIT_ASSERT(!(small >= med));

   CPPUNIT_ASSERT(!(med <= small));
   CPPUNIT_ASSERT(!(med == small));
   CPPUNIT_ASSERT(med != small);
   CPPUNIT_ASSERT(med > small);
   CPPUNIT_ASSERT(med >= small);

   CPPUNIT_ASSERT(med2 == med);
   CPPUNIT_ASSERT(med2 <= med);
   CPPUNIT_ASSERT(med2 >= med);
   CPPUNIT_ASSERT(!(med2 != med));
   CPPUNIT_ASSERT(!(med2 < med));
   CPPUNIT_ASSERT(!(med2 > med));
   
   CPPUNIT_ASSERT(med < large);
   CPPUNIT_ASSERT(med <= large);
   CPPUNIT_ASSERT(!(med == large));
   CPPUNIT_ASSERT(med != large);
   CPPUNIT_ASSERT(!(med > large));
   CPPUNIT_ASSERT(!(med >= large));

   CPPUNIT_ASSERT(large < largest);
   CPPUNIT_ASSERT(!(largest < large));
   CPPUNIT_ASSERT(large <= largest);
   CPPUNIT_ASSERT(!(large == largest));
   CPPUNIT_ASSERT(large != largest);
   CPPUNIT_ASSERT(!(large > largest));
   CPPUNIT_ASSERT(!(large >= largest));

   CPPUNIT_ASSERT(larger < largest);
   CPPUNIT_ASSERT(!(largest < larger));
}

void testEventID::iterationTest()
{
   EventID first = EventID::firstValidEvent();
   
   EventID second = first.next(1);
   CPPUNIT_ASSERT(first < second);
   CPPUNIT_ASSERT(first == (second.previous(1)));
   
   EventID run2(2, 1, 0);
   CPPUNIT_ASSERT(run2 < run2.nextRun(1));
   CPPUNIT_ASSERT(run2 > run2.previousRunLastEvent(1));
   CPPUNIT_ASSERT(first < run2.previousRunLastEvent(1));
   CPPUNIT_ASSERT(run2 < first.nextRunFirstEvent(1));
   
   EventID run2Last(2, 1, EventID::maxEventNumber());
   CPPUNIT_ASSERT(run2Last.next(1) == run2Last.nextRunFirstEvent(1));
}
