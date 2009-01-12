/*
 *  lumirange_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Eric Vaandering December 2008.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

using namespace edm;

class testLumiRange: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testLumiRange);

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
CPPUNIT_TEST_SUITE_REGISTRATION(testLumiRange);


void testLumiRange::constructTest()
{
   const RunNumber_t rb = 1;
   const RunNumber_t re = 2;
   const LuminosityBlockNumber_t lb = 3;
   const LuminosityBlockNumber_t le = 4;

   LuminosityBlockRange normal(rb, lb, re, le);
   LuminosityBlockRange maxed(rb, 0, re, 0);
   LuminosityBlockID    dummy;

   CPPUNIT_ASSERT(normal.startRun() == rb);
   CPPUNIT_ASSERT(normal.endRun()   == re);
   CPPUNIT_ASSERT(normal.startLumi() == lb);
   CPPUNIT_ASSERT(normal.endLumi()   == le);
   CPPUNIT_ASSERT(maxed.startLumiID().luminosityBlock() == dummy.maxLuminosityBlockNumber());
   CPPUNIT_ASSERT(maxed.endLumiID().luminosityBlock() == dummy.maxLuminosityBlockNumber());

}

void testLumiRange::comparisonTest()
{
    const LuminosityBlockID small(1,1);
    const LuminosityBlockID med(7, 2);
    const LuminosityBlockID large(8,10);
    const LuminosityBlockID larger(10,1);
    const LuminosityBlockRange normal(5,1,8,1);
    const LuminosityBlockRange maxed(5,1,8,0);

    CPPUNIT_ASSERT(!normal.contains(small));
    CPPUNIT_ASSERT(normal.contains(med));
    CPPUNIT_ASSERT(!normal.contains(large));
    CPPUNIT_ASSERT(!normal.contains(larger));

    CPPUNIT_ASSERT(!maxed.contains(small));
    CPPUNIT_ASSERT(maxed.contains(med));
    CPPUNIT_ASSERT(maxed.contains(large));
    CPPUNIT_ASSERT(!maxed.contains(larger));

}

// void testEventID::iterationTest()
// {
//    EventID first = EventID::firstValidEvent();
//
//    EventID second = first.next();
//    CPPUNIT_ASSERT(first < second);
//    CPPUNIT_ASSERT(first == (second.previous()));
//
//    EventID run2(2, 0);
//    CPPUNIT_ASSERT(run2 < run2.nextRun());
//    CPPUNIT_ASSERT(run2 > run2.previousRunLastEvent());
//    CPPUNIT_ASSERT(first < run2.previousRunLastEvent());
//    CPPUNIT_ASSERT(run2 < first.nextRunFirstEvent());
//
//    EventID run2Last(2, EventID::maxEventNumber());
//    CPPUNIT_ASSERT(run2Last.next() == run2Last.nextRunFirstEvent());
// }
