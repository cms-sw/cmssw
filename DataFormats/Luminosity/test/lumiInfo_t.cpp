#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/LumiInfo.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class TestLumiInfo: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestLumiInfo);  
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST(testFill);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void setUp() {}
  void tearDown() {}

  void testConstructor();
  void testFill();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestLumiInfo);

void
TestLumiInfo::testConstructor() {
  std::cout << "\nTesting LumiInfo\n";
  LumiInfo lumiInfo;
  
  lumiInfo.setDeadFraction(0.4);
  CPPUNIT_ASSERT(fabs(lumiInfo.deadFraction() - 0.4) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.liveFraction() - 0.6) < 1e-5);
}

void
TestLumiInfo::testFill() {
  LumiInfo lumiInfo;
  lumiInfo.setDeadFraction(0.5);
  std::vector<float> lumBX;
  lumBX.push_back(1.0f);
  lumBX.push_back(2.0f);
  lumBX.push_back(3.0f);

  std::vector<float> beam1;
  beam1.push_back(4.0f);
  beam1.push_back(5.0f);
  beam1.push_back(6.0f);

  std::vector<float> beam2;
  beam2.push_back(7.0f);
  beam2.push_back(8.0f);
  beam2.push_back(9.0f);

  lumiInfo.fill(lumBX, beam1, beam2);

  CPPUNIT_ASSERT(fabs(lumiInfo.getInstLumiBX(0) - 1.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getInstLumiBX(1) - 2.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getInstLumiBX(2) - 3.0f) < 1e-5);

  CPPUNIT_ASSERT(fabs(lumiInfo.instLuminosity() - 6.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.integLuminosity() - 6.0f*lumiInfo.lumiSectionLength()) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.recordedLuminosity() - 3.0f*lumiInfo.lumiSectionLength()) < 1e-5);
  
  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam1IntensityBX(0) - 4.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam1IntensityBX(1) - 5.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam1IntensityBX(2) - 6.0f) < 1e-5);

  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam2IntensityBX(0) - 7.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam2IntensityBX(1) - 8.0f) < 1e-5);
  CPPUNIT_ASSERT(fabs(lumiInfo.getBeam2IntensityBX(2) - 9.0f) < 1e-5);

  CPPUNIT_ASSERT(lumiInfo.isProductEqual(lumiInfo));

  LumiInfo lumiInfo2;
  CPPUNIT_ASSERT(!lumiInfo.isProductEqual(lumiInfo2));

  std::cout << lumiInfo;
}
