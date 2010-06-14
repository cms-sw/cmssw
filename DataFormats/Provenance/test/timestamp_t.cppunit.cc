/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Provenance/interface/Timestamp.h"

using namespace edm;

class testTimestamp: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testTimestamp);
   
   CPPUNIT_TEST(constructTest);
   CPPUNIT_TEST(comparisonTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void constructTest();
   void comparisonTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTimestamp);


void testTimestamp::constructTest()
{
   const TimeValue_t t = 2;

   Timestamp temp(t);
   
   CPPUNIT_ASSERT(temp.value() == t);
   
   CPPUNIT_ASSERT(Timestamp::invalidTimestamp() < Timestamp::beginOfTime());
   CPPUNIT_ASSERT(Timestamp::beginOfTime() < Timestamp::endOfTime());
}

void testTimestamp::comparisonTest()
{
   const Timestamp small(1);
   const Timestamp med(2);
   
   CPPUNIT_ASSERT(small < med);
   CPPUNIT_ASSERT(small <= med);
   CPPUNIT_ASSERT(!(small == med));
   CPPUNIT_ASSERT(small != med);
   CPPUNIT_ASSERT(!(small > med));
   CPPUNIT_ASSERT(!(small >= med));

}
