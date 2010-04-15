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

#define private public
#define protected public
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#undef private
#undef protected

#ifndef TestSiStripGain_cc
#define TestSiStripGain_cc

class TestSiStripGain : public CppUnit::TestFixture {
public:
  TestSiStripGain() {}
  void setUp()
  {
    detId = 436282904;

    apvGain1 = new SiStripApvGain;
    std::vector<float> theSiStripVector;
    theSiStripVector.push_back(1.);
    theSiStripVector.push_back(0.8);
    theSiStripVector.push_back(1.2);
    theSiStripVector.push_back(2.);
    fillApvGain(apvGain1, detId, theSiStripVector);

    apvGain2 = new SiStripApvGain;
    theSiStripVector.clear();
    theSiStripVector.push_back(1.);
    theSiStripVector.push_back(1./0.8);
    theSiStripVector.push_back(1./1.2);
    theSiStripVector.push_back(2.);
    fillApvGain(apvGain2, detId, theSiStripVector);
  }

  void tearDown()
  {
    delete apvGain1;
    delete apvGain2;
  }

  void testConstructor()
  {
    // Test with normalization factor = 1
    apvGainsTest(1.);
    // Test with normalization factor != 1
    apvGainsTest(2.);
  }

  void testMultiply()
  {
    // Test with norm = 1
    multiplyTest(1., 1.);

    // Test with norm != 1
    multiplyTest(2., 3.);
  }

  void multiplyTest(const float & norm1, const float & norm2)
  {
    SiStripGain gain(*apvGain1, norm1);
    gain.multiply(*apvGain2, norm2);
    SiStripApvGain::Range range = gain.getRange(detId);

    CPPUNIT_ASSERT( float(gain.getApvGain(0, range)) == float(1./norm1*1./norm2));
    CPPUNIT_ASSERT( float(gain.getApvGain(1, range)) == float(0.8/norm1*1./(0.8*norm2)));
    CPPUNIT_ASSERT( float(gain.getApvGain(2, range)) == float(1.2/norm1*1./(1.2*norm2)));
    CPPUNIT_ASSERT( float(gain.getApvGain(3, range)) == float(2./norm1*2./norm2));
  }

  void apvGainsTest(const float & norm)
  {
    SiStripGain gain(*apvGain1, norm);
    SiStripApvGain::Range range = gain.getRange(detId);
    CPPUNIT_ASSERT( float(gain.getApvGain(0, range)) == float(1./norm));
    CPPUNIT_ASSERT( float(gain.getApvGain(1, range)) == float(0.8/norm));
    CPPUNIT_ASSERT( float(gain.getApvGain(2, range)) == float(1.2/norm));
    CPPUNIT_ASSERT( float(gain.getApvGain(3, range)) == float(2./norm));

    SiStripGain gain2(*apvGain2, norm);
    SiStripApvGain::Range range2 = gain2.getRange(detId);
    CPPUNIT_ASSERT( float(gain2.getApvGain(0, range2)) == float(1./norm));
    CPPUNIT_ASSERT( float(gain2.getApvGain(1, range2)) == float(1./(norm*0.8)));
    CPPUNIT_ASSERT( float(gain2.getApvGain(2, range2)) == float(1./(norm*1.2)));
    CPPUNIT_ASSERT( float(gain2.getApvGain(3, range2)) == float(2./norm));
  }

  void fillApvGain(SiStripApvGain * apvGain, const uint32_t detId, const std::vector<float> & theSiStripVector)
  {
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    apvGain->put(detId,range);
  }

  SiStripApvGain * apvGain1;
  SiStripApvGain * apvGain2;
  uint32_t detId;

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestSiStripGain );
  CPPUNIT_TEST( testConstructor );
  CPPUNIT_TEST( testMultiply );
  CPPUNIT_TEST_SUITE_END();
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripGain );

#endif
