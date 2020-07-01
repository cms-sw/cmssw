#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class TestBeamCurrentInfo : public CppUnit::TestFixture {
  static const float tol;

  CPPUNIT_TEST_SUITE(TestBeamCurrentInfo);
  CPPUNIT_TEST(testFill);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testFill();
};

// tolerance (relative)
const float TestBeamCurrentInfo::tol = 1e-3;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestBeamCurrentInfo);

void TestBeamCurrentInfo::testFill() {
  BeamCurrentInfo beamCurrentInfo;

  // Use somewhat realistic data so that we don't end up out-of-range.

  std::vector<float> beam1;
  beam1.push_back(0.042e10f);
  beam1.push_back(14.481e10f);
  beam1.push_back(0.422e10f);

  std::vector<float> beam2;
  beam2.push_back(0.081e10f);
  beam2.push_back(14.662e10f);
  beam2.push_back(0.135e10f);

  beamCurrentInfo.fill(beam1, beam2);

  std::cout << beamCurrentInfo;

  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(0) - beam1[0]) <
                 beamCurrentInfo.getBeam1IntensityBX(0) * tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(1) - beam1[1]) <
                 beamCurrentInfo.getBeam1IntensityBX(1) * tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam1IntensityBX(2) - beam1[2]) <
                 beamCurrentInfo.getBeam1IntensityBX(2) * tol);

  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(0) - beam2[0]) <
                 beamCurrentInfo.getBeam2IntensityBX(0) * tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(1) - beam2[1]) <
                 beamCurrentInfo.getBeam2IntensityBX(1) * tol);
  CPPUNIT_ASSERT(std::abs(beamCurrentInfo.getBeam2IntensityBX(2) - beam2[2]) <
                 beamCurrentInfo.getBeam2IntensityBX(2) * tol);

  CPPUNIT_ASSERT(beamCurrentInfo.isProductEqual(beamCurrentInfo));

  BeamCurrentInfo beamCurrentInfo2;
  CPPUNIT_ASSERT(!beamCurrentInfo.isProductEqual(beamCurrentInfo2));
}
