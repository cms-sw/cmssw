#include <iomanip>

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace geant_units;
using namespace geant_units::operators;
using namespace std;

class testDDUnits : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDUnits);
  CPPUNIT_TEST(checkUnits);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkUnits();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDUnits);

void testDDUnits::checkUnits() {
  std::cout << "\nMy pi: " << std::setprecision(16) << piRadians << " == " << 1_pi << " == " << 1._pi << "\n";

  CPPUNIT_ASSERT(M_PI == piRadians);
  CPPUNIT_ASSERT(M_PI == 1_pi);
  CPPUNIT_ASSERT(M_PI == 1._pi);

  double twoPiAngle = 2_pi;
  std::cout << "My 2pi angle: " << twoPiAngle << " [rad] == " << convertRadToDeg(twoPiAngle) << " [deg]\n";

  CPPUNIT_ASSERT(2 * M_PI == 2 * piRadians);
  CPPUNIT_ASSERT(2 * M_PI == twoPiAngle);
  CPPUNIT_ASSERT(2 * M_PI == 2_pi);
  CPPUNIT_ASSERT(2 * M_PI == 2._pi);

  CPPUNIT_ASSERT(90_deg == 1_pi / 2);
  CPPUNIT_ASSERT(120_deg == 2_pi / 3);
  CPPUNIT_ASSERT(135_deg == 3_pi / 4);
  CPPUNIT_ASSERT(150_deg == 5_pi / 6);

  double angle90 = convertRadToDeg(1_pi / 2);
  CPPUNIT_ASSERT(angle90 == 90);

  double angle120 = convertRadToDeg(2_pi / 3);
  CPPUNIT_ASSERT(almostEqual(angle120, 120., 2));

  double angle135 = convertRadToDeg(3_pi / 4);
  CPPUNIT_ASSERT(almostEqual(angle135, 135., 2));

  double angle150 = convertRadToDeg(5_pi / 6);
  CPPUNIT_ASSERT(almostEqual(angle150, 150., 2));

  cout << "Mass of 1 kg is " << 1._kg << " or " << 1 * CLHEP::kg << "\n";
  cout << "Mass of 1  g is " << 1._g << " or " << 1 * CLHEP::g << "\n";
  cout << "Ratio of 1._kg / CLHEP::kg is " << 1._kg / (1 * CLHEP::kg) << endl;
  cout << "Difference of 1._kg - CLHEP::kg is " << std::abs(1._kg - (1 * CLHEP::kg)) << endl;
  cout << "Ratio of 1._g / CLHEP::g is " << 1._g / (1 * CLHEP::g) << endl;
  cout << "Difference of 1._g - CLHEP::g is " << std::abs(1._g - (1 * CLHEP::g)) << endl;
}
