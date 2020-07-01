#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/LumiInfo.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class TestLumiInfo : public CppUnit::TestFixture {
  static const float tol;

  CPPUNIT_TEST_SUITE(TestLumiInfo);
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST(testFill);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testConstructor();
  void testFill();
};

const float TestLumiInfo::tol = 1e-5;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestLumiInfo);

void TestLumiInfo::testConstructor() {
  std::cout << "\nTesting LumiInfo\n";
  LumiInfo lumiInfo;

  lumiInfo.setDeadFraction(0.4);
  CPPUNIT_ASSERT(std::abs(lumiInfo.getDeadFraction() - 0.4) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.getLiveFraction() - 0.6) < tol);
}

void TestLumiInfo::testFill() {
  LumiInfo lumiInfo;
  lumiInfo.setDeadFraction(0.5);
  std::vector<float> lumBX;
  lumBX.push_back(1.0f);
  lumBX.push_back(2.0f);
  lumBX.push_back(3.0f);

  lumiInfo.setInstLumiAllBX(lumBX);
  lumiInfo.setTotalInstLumi(8.0f);

  CPPUNIT_ASSERT(std::abs(lumiInfo.getInstLumiBX(0) - 1.0f) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.getInstLumiBX(1) - 2.0f) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.getInstLumiBX(2) - 3.0f) < tol);

  CPPUNIT_ASSERT(std::abs(lumiInfo.getTotalInstLumi() - 8.0f) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.instLuminosityBXSum() - 6.0f) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.integLuminosity() - 8.0f * lumiInfo.lumiSectionLength()) < tol);
  CPPUNIT_ASSERT(std::abs(lumiInfo.recordedLuminosity() - 4.0f * lumiInfo.lumiSectionLength()) < tol);

  lumiInfo.setTotalInstToBXSum();
  CPPUNIT_ASSERT(std::abs(lumiInfo.getTotalInstLumi() - 6.0f) < tol);

  CPPUNIT_ASSERT(lumiInfo.isProductEqual(lumiInfo));

  LumiInfo lumiInfo2;
  CPPUNIT_ASSERT(!lumiInfo.isProductEqual(lumiInfo2));

  std::cout << lumiInfo;
}
