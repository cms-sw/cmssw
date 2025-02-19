/*
 *  interval_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/30/05.
 *  Changed by Viji Sundararajan on 29-Jun-05.
 *
 */

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include <cppunit/extensions/HelperMacros.h>

using edm::Timestamp;
using edm::IOVSyncValue;
using edm::ValidityInterval;

class testinterval: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testinterval);

CPPUNIT_TEST(comparisonTest);
CPPUNIT_TEST(timestampAssignmentTest);
CPPUNIT_TEST(intervalAssignmentTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void comparisonTest();
  void timestampAssignmentTest();
  void intervalAssignmentTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testinterval);


void testinterval::comparisonTest()
{
   const IOVSyncValue invalid(IOVSyncValue::invalidIOVSyncValue());
   
   const Timestamp t_1(1);
   const IOVSyncValue one(t_1);
   const Timestamp t_2(2);
   const IOVSyncValue two(t_2);
   
   CPPUNIT_ASSERT(invalid == IOVSyncValue::invalidIOVSyncValue());
   CPPUNIT_ASSERT(one == IOVSyncValue(t_1));
   
   CPPUNIT_ASSERT(invalid != one);

   CPPUNIT_ASSERT(one < two);
   CPPUNIT_ASSERT(!(one > two));
   CPPUNIT_ASSERT(two > one);
   CPPUNIT_ASSERT(!(two < one));
   
   CPPUNIT_ASSERT(one != two);
   CPPUNIT_ASSERT(! (one == two));

   CPPUNIT_ASSERT(one <= two);
   CPPUNIT_ASSERT(one <= one);
   CPPUNIT_ASSERT(one >= one);
   CPPUNIT_ASSERT(!(one >= two));
}

void testinterval::timestampAssignmentTest()
{
   const Timestamp t_1(1);
   const IOVSyncValue one(t_1);
   
   IOVSyncValue temp(IOVSyncValue::invalidIOVSyncValue());
   CPPUNIT_ASSERT(temp != one);
   temp = one;
   CPPUNIT_ASSERT(temp == one);
}

void testinterval::intervalAssignmentTest()
{
   ValidityInterval temp;
   const Timestamp t_1(1);
   const IOVSyncValue s_1(t_1);
   const ValidityInterval oneAndTwo(s_1, IOVSyncValue(Timestamp(2)));
   
   CPPUNIT_ASSERT(temp != oneAndTwo);
   CPPUNIT_ASSERT(! (temp == oneAndTwo));
   
   temp = oneAndTwo;
   CPPUNIT_ASSERT(temp == oneAndTwo);
}
