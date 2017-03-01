/*
 *  eventid_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;

class testIOVSyncValue: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testIOVSyncValue);
   
   CPPUNIT_TEST(constructTest);
   CPPUNIT_TEST(constructTimeTest);
   CPPUNIT_TEST(comparisonTest);
   CPPUNIT_TEST(comparisonTimeTest);
   CPPUNIT_TEST(invalidComparisonTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void constructTest();
   void comparisonTest();
   void constructTimeTest();
   void comparisonTimeTest();
   void invalidComparisonTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testIOVSyncValue);


void testIOVSyncValue::constructTest()
{
   {
      const EventID t(2,0,0);
      
      IOVSyncValue temp(t);
      
      CPPUNIT_ASSERT(temp.eventID() == t);
      
      
      CPPUNIT_ASSERT(IOVSyncValue::invalidIOVSyncValue() != temp);
      CPPUNIT_ASSERT(!(IOVSyncValue::invalidIOVSyncValue() == temp));
      CPPUNIT_ASSERT(IOVSyncValue::beginOfTime() < temp);
      CPPUNIT_ASSERT(IOVSyncValue::endOfTime() > temp);
   }
   CPPUNIT_ASSERT(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
   CPPUNIT_ASSERT(IOVSyncValue::beginOfTime() < IOVSyncValue::endOfTime());

   {
      const EventID t(2,3,1);
      
      IOVSyncValue temp(t);
      
      CPPUNIT_ASSERT(temp.eventID() == t);
      CPPUNIT_ASSERT(temp.luminosityBlockNumber() == 3);
      
      CPPUNIT_ASSERT(IOVSyncValue::invalidIOVSyncValue() != temp);
      CPPUNIT_ASSERT(!(IOVSyncValue::invalidIOVSyncValue() == temp));
      CPPUNIT_ASSERT(IOVSyncValue::beginOfTime() < temp);
      CPPUNIT_ASSERT(IOVSyncValue::endOfTime() > temp);
   }
   
}

void testIOVSyncValue::constructTimeTest()
{
   const Timestamp t(2);
   
   IOVSyncValue temp(t);
   
   CPPUNIT_ASSERT(temp.time() == t);
   
   CPPUNIT_ASSERT(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
   CPPUNIT_ASSERT(IOVSyncValue::beginOfTime() < IOVSyncValue::endOfTime());
   
   CPPUNIT_ASSERT(IOVSyncValue::invalidIOVSyncValue() != temp);
   CPPUNIT_ASSERT(!(IOVSyncValue::invalidIOVSyncValue() == temp));
   CPPUNIT_ASSERT(IOVSyncValue::beginOfTime() < temp);
   CPPUNIT_ASSERT(IOVSyncValue::endOfTime() > temp);
}

void testIOVSyncValue::comparisonTest()
{
   {
      const IOVSyncValue small(EventID(1,1,1));
      const IOVSyncValue med(EventID(2,1,2));
      
      CPPUNIT_ASSERT(small.comparable(med));
      CPPUNIT_ASSERT(small < med);
      CPPUNIT_ASSERT(small <= med);
      CPPUNIT_ASSERT(!(small == med));
      CPPUNIT_ASSERT(small != med);
      CPPUNIT_ASSERT(!(small > med));
      CPPUNIT_ASSERT(!(small >= med));
   }
   {
      const IOVSyncValue small(EventID(2,1,1));
      const IOVSyncValue med(EventID(2,1,2));
      
      CPPUNIT_ASSERT(small < med);
      CPPUNIT_ASSERT(small <= med);
      CPPUNIT_ASSERT(!(small == med));
      CPPUNIT_ASSERT(small != med);
      CPPUNIT_ASSERT(!(small > med));
      CPPUNIT_ASSERT(!(small >= med));
   }
   {
      const IOVSyncValue small(EventID(2,1,2));
      const IOVSyncValue med(EventID(3,1,1));
      
      CPPUNIT_ASSERT(small < med);
      CPPUNIT_ASSERT(small <= med);
      CPPUNIT_ASSERT(!(small == med));
      CPPUNIT_ASSERT(small != med);
      CPPUNIT_ASSERT(!(small > med));
      CPPUNIT_ASSERT(!(small >= med));
   }
   {
      const IOVSyncValue small(EventID(2,2,1));
      const IOVSyncValue med(EventID(2,2,2));
      
      CPPUNIT_ASSERT(small < med);
      CPPUNIT_ASSERT(small <= med);
      CPPUNIT_ASSERT(!(small == med));
      CPPUNIT_ASSERT(small != med);
      CPPUNIT_ASSERT(!(small > med));
      CPPUNIT_ASSERT(!(small >= med));
   }
   {
      const IOVSyncValue small(EventID(2,1,3));
      const IOVSyncValue med(EventID(2,2,2));
      
      CPPUNIT_ASSERT(small < med);
      CPPUNIT_ASSERT(small <= med);
      CPPUNIT_ASSERT(!(small == med));
      CPPUNIT_ASSERT(small != med);
      CPPUNIT_ASSERT(!(small > med));
      CPPUNIT_ASSERT(!(small >= med));
   }
}


void testIOVSyncValue::comparisonTimeTest()
{
   const IOVSyncValue small(Timestamp(1));
   const IOVSyncValue med(Timestamp(2));

   CPPUNIT_ASSERT(small.comparable(med));
   CPPUNIT_ASSERT(small < med);
   CPPUNIT_ASSERT(small <= med);
   CPPUNIT_ASSERT(!(small == med));
   CPPUNIT_ASSERT(small != med);
   CPPUNIT_ASSERT(!(small > med));
   CPPUNIT_ASSERT(!(small >= med));

}

void 
testIOVSyncValue::invalidComparisonTest()
{
  const IOVSyncValue timeBased(Timestamp(1));
  const IOVSyncValue eventBased(EventID(3,2,1));

  CPPUNIT_ASSERT(! timeBased.comparable(eventBased));
  CPPUNIT_ASSERT(! eventBased.comparable(timeBased));
  CPPUNIT_ASSERT_THROW( [&](){return timeBased < eventBased;}()  , cms::Exception);
  CPPUNIT_ASSERT_THROW( [&](){return timeBased <= eventBased;}() , cms::Exception);
  CPPUNIT_ASSERT( !(timeBased == eventBased));
  CPPUNIT_ASSERT( timeBased != eventBased);
  CPPUNIT_ASSERT_THROW( [&]() {return timeBased > eventBased;}()  , cms::Exception);
  CPPUNIT_ASSERT_THROW( [&]() {return timeBased >= eventBased;}() , cms::Exception);
}
