/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

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
   const EventNumber_t et = 1;
   const RunNumber_t rt = 2;

   EventID temp(rt, et);
   
   CPPUNIT_ASSERT(temp.run() == rt);
   CPPUNIT_ASSERT(temp.event() == et);
}

void testEventID::comparisonTest()
{
   const EventID small(1,1);
   const EventID med(2, 2);
   const EventID med2(2,2);
   const EventID large(3, 2);
   const EventID largest(3,3);
   
   CPPUNIT_ASSERT(small < med);
   CPPUNIT_ASSERT(small <= med);
   CPPUNIT_ASSERT(!(small == med));
   CPPUNIT_ASSERT(small != med);
   CPPUNIT_ASSERT(!(small > med));
   CPPUNIT_ASSERT(!(small >= med));

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
   CPPUNIT_ASSERT(large <= largest);
   CPPUNIT_ASSERT(!(large == largest));
   CPPUNIT_ASSERT(large != largest);
   CPPUNIT_ASSERT(!(large > largest));
   CPPUNIT_ASSERT(!(large >= largest));
   
}

void testEventID::iterationTest()
{
   EventID first = EventID::firstValidEvent();
   
   EventID second = first.next();
   CPPUNIT_ASSERT(first < second);
   CPPUNIT_ASSERT(first == (second.previous()));
   
   EventID run2(2, 0);
   CPPUNIT_ASSERT(run2 < run2.nextRun());
   CPPUNIT_ASSERT(run2 > run2.previousRunLastEvent());
   CPPUNIT_ASSERT(first < run2.previousRunLastEvent());
   CPPUNIT_ASSERT(run2 < first.nextRunFirstEvent());
   
   EventID run2Last(2, EventID::maxEventNumber());
   CPPUNIT_ASSERT(run2Last.next() == run2Last.nextRunFirstEvent());
}
