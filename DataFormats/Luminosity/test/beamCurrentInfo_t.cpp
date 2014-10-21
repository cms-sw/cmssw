#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class TestBeamCurrentInfo: public CppUnit::TestFixture
{
  static const float tol;

  CPPUNIT_TEST_SUITE(TestBeamCurrentInfo);  
  CPPUNIT_TEST(testFill);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void setUp() {}
  void tearDown() {}

  void testFill();
};

const float TestBeamCurrentInfo::tol = 1e-5;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestBeamCurrentInfo);

void
TestBeamCurrentInfo::testFill() {
  BeamCurrentInfo beamCurrentInfo;

  std::vector<float> beam1;
  beam1.push_back(4.0f);
  beam1.push_back(5.0f);
  beam1.push_back(6.0f);

  std::vector<float> beam2;
  beam2.push_back(7.0f);
  beam2.push_back(8.0f);
  beam2.push_back(9.0f);

  beamCurrentInfo.fill(beam1, beam2);

  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(0) - 4.0f) < tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(1) - 5.0f) < tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(2) - 6.0f) < tol);

  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(0) - 7.0f) < tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(1) - 8.0f) < tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(2) - 9.0f) < tol);

  CPPUNIT_ASSERT(beamCurrentInfo.isProductEqual(beamCurrentInfo));

  BeamCurrentInfo beamCurrentInfo2;
  CPPUNIT_ASSERT(!beamCurrentInfo.isProductEqual(beamCurrentInfo2));

  std::cout << beamCurrentInfo;
}
