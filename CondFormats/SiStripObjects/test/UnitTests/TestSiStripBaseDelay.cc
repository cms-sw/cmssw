#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"

class TestSiStripBaseDelay : public CppUnit::TestFixture
{
 public:
  TestSiStripBaseDelay() {}

  void setUp()
  {
    totDelays = 4;

    detId.push_back(0);
    detId.push_back(1);
    detId.push_back(2);
    detId.push_back(3);

    coarseDelay.push_back(0);
    coarseDelay.push_back(3);
    coarseDelay.push_back(1);
    coarseDelay.push_back(2);

    fineDelay = coarseDelay;

    for( unsigned int i=0; i<totDelays; ++i ) {
      delay.put(detId[i], coarseDelay[i], fineDelay[i]);
    }
  }

  void tearDown() {}

  void testDelays()
  {
    std::vector<SiStripBaseDelay::Delay> delays;
    delay.delays(delays);
    CPPUNIT_ASSERT( delays.size() == totDelays );
  }

  void testCoarseDelay()
  {
    for( unsigned int i=0; i<totDelays; ++i ) {
      CPPUNIT_ASSERT( delay.coarseDelay(i) == coarseDelay[i] );
    }
  }

  void testFineDelay()
  {
    for( unsigned int i=0; i<totDelays; ++i ) {
      CPPUNIT_ASSERT( delay.fineDelay(i) == fineDelay[i] );
    }
  }

  void testDelay()
  {
    for( unsigned int i=0; i<totDelays; ++i ) {
      CPPUNIT_ASSERT( delay.delay(i) == coarseDelay[i]*25 + fineDelay[i]*(25/24.) );
    }
  }

  CPPUNIT_TEST_SUITE( TestSiStripBaseDelay );
  CPPUNIT_TEST( testDelays );
  CPPUNIT_TEST( testCoarseDelay );
  CPPUNIT_TEST( testFineDelay );
  CPPUNIT_TEST( testDelay );
  CPPUNIT_TEST_SUITE_END();

  SiStripBaseDelay delay;
  unsigned int totDelays;
  std::vector<uint32_t> detId;
  std::vector<uint16_t> coarseDelay;
  std::vector<uint16_t> fineDelay;
};

CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripBaseDelay );
