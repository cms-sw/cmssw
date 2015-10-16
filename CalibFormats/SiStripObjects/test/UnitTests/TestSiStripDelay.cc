#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include <algorithm>
#include <iterator>
#include <boost/foreach.hpp>
#include <cassert>

#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"

#ifndef TestSiStripDelay_cc
#define TestSiStripDelay_cc

class TestSiStripDelay : public CppUnit::TestFixture {
public:
  TestSiStripDelay() {}
  void setUp()
  {
    baseDelay1 = new SiStripBaseDelay;
    baseDelaySign1 = 1;
    baseDelay2 = new SiStripBaseDelay;
    baseDelaySign2 = -1;
    totalDelays = 15000;    
    // totalDelays = 1;
    for( uint32_t i=0; i<totalDelays; ++i ) {
      detIds.push_back(i);
      coarseDelay.push_back(i/1000);
      fineDelay.push_back(i/2000);
      baseDelay1->put(i, i/1000, i/2000);
      baseDelay2->put(i, i/2000, i/1000);
      // coarseDelay.push_back(1);
      // fineDelay.push_back(2);
      // baseDelay1->put(i, 1, 2);
      // baseDelay2->put(i, 2, 1);
    }
  }

  void tearDown()
  {
    delete baseDelay1;
    delete baseDelay2;
  }

  void testConstructor()
  {
    SiStripDelay delay(*baseDelay1, baseDelaySign1, std::make_pair("baseDelay1", "delay1"));
    CPPUNIT_ASSERT( delay.getNumberOfTags() == 1 );
    CPPUNIT_ASSERT( delay.getBaseDelay(0) == baseDelay1 );
    CPPUNIT_ASSERT( delay.getTagSign(0) == 1 );
    CPPUNIT_ASSERT( delay.getRcdName(0) == "baseDelay1" );
    CPPUNIT_ASSERT( delay.getLabelName(0) == "delay1" );
  }

  void testFillNewDelay()
  {
    SiStripDelay delay(*baseDelay1, baseDelaySign1, std::make_pair("baseDelay1", "delay1"));
    delay.fillNewDelay(*baseDelay2, baseDelaySign2, std::make_pair("baseDelay2", "delay2"));
    CPPUNIT_ASSERT( delay.getNumberOfTags() == 2 );
    CPPUNIT_ASSERT( delay.getBaseDelay(0) == baseDelay1 );
    CPPUNIT_ASSERT( delay.getTagSign(0) == baseDelaySign1 );
    CPPUNIT_ASSERT( delay.getRcdName(0) == "baseDelay1" );
    CPPUNIT_ASSERT( delay.getLabelName(0) == "delay1" );
    CPPUNIT_ASSERT( delay.getBaseDelay(1) == baseDelay2 );
    CPPUNIT_ASSERT( delay.getTagSign(1) == baseDelaySign2 );
    CPPUNIT_ASSERT( delay.getRcdName(1) == "baseDelay2" );
    CPPUNIT_ASSERT( delay.getLabelName(1) == "delay2" );
  }

  void testMakeDelay()
  {
    SiStripDelay delay(*baseDelay1, baseDelaySign1, std::make_pair("baseDelay1", "delay1"));    
    CPPUNIT_ASSERT(delay.makeDelay());
    for( uint32_t i=0; i<totalDelays; ++i ) {
      CPPUNIT_ASSERT( float(delay.getDelay(detIds[i])) == float(baseDelay1->delay(detIds[i])*baseDelaySign1) );
    }
    delay.fillNewDelay(*baseDelay2, baseDelaySign2, std::make_pair("baseDelay2", "delay2"));
    CPPUNIT_ASSERT(delay.makeDelay());
    for( uint32_t i=0; i<totalDelays; ++i ) {
//       std::cout << "float(delay.getDelay("<<detIds[i]<<")) = " << float(delay.getDelay(detIds[i])) << std::endl;
//       std::cout << "float(baseDelay1->delay("<<detIds[i]<<") = " << float(baseDelay1->delay(detIds[i])) << std::endl;
//       std::cout << "float(baseDelay2->delay("<<detIds[i]<<") = " << float(baseDelay2->delay(detIds[i])) << std::endl;
//       std::cout << "float(baseDelay1->delay("<<detIds[i]<<")*"<<baseDelaySign1<<" - baseDelay2->delay("<<detIds[i]<<"))*"<<baseDelaySign2<<" = " << float(baseDelay1->delay(detIds[i])*baseDelaySign1 + baseDelay2->delay(detIds[i])*baseDelaySign2) << std::endl;
      CPPUNIT_ASSERT( float(delay.getDelay(detIds[i])) == float(baseDelay1->delay(detIds[i])*baseDelaySign1 + baseDelay2->delay(detIds[i])*baseDelaySign2) );
    }
  }

  SiStripBaseDelay * baseDelay1;
  int baseDelaySign1;
  SiStripBaseDelay * baseDelay2;
  int baseDelaySign2;
  std::vector<uint32_t> detIds;
  std::vector<uint16_t> coarseDelay;
  std::vector<uint16_t> fineDelay;
  uint32_t totalDelays;

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestSiStripDelay );
  CPPUNIT_TEST( testConstructor );
  CPPUNIT_TEST( testFillNewDelay );
  CPPUNIT_TEST( testMakeDelay );
  CPPUNIT_TEST_SUITE_END();
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripDelay );

#endif
