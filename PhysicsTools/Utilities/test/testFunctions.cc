#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Identity.h"
#include "PhysicsTools/Utilities/interface/Composition.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include <cmath>

class testFunctions : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFunctions);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testFunctions);

void testFunctions::checkAll() {
  using namespace funct;
  Gaussian g1(0, 1);
  Gaussian g2(1, 1);
  Identity i;
  const double epsilon = 1.e-6;
  Sum<Gaussian, Gaussian>::type g1plus2 = g1 + g2;
  Product<Gaussian, Gaussian>::type g1times2 = g1 * g2; 
  Difference<Gaussian, Gaussian>::type g1minus2 = g1 - g2;
  Ratio<Gaussian, Gaussian>::type g1over2 = g1 / g2; 
  Minus<Gaussian>::type gm1 = - g1;
  Composition<Identity, Gaussian>::type gg1 = compose(i, g1);
  double x = 0.5;
  CPPUNIT_ASSERT(fabs(g1plus2(x) - (g1(x) + g2(x))) < epsilon);
  CPPUNIT_ASSERT(fabs(g1times2(x) - (g1(x) * g2(x))) < epsilon);
  CPPUNIT_ASSERT(fabs(g1minus2(x) - (g1(x) - g2(x))) < epsilon);
  CPPUNIT_ASSERT(fabs(g1over2(x) - (g1(x) / g2(x))) < epsilon);
  CPPUNIT_ASSERT(fabs(gm1(x) - (-g1(x)) < epsilon));
  Convolution<Gaussian, Gaussian>::type gg(g1, g1, -5, 5, 1000);
  CPPUNIT_ASSERT(fabs(gg(0) - g1(0)/sqrt(2))<epsilon);
  CPPUNIT_ASSERT(gg1(0) == g1(0));
}
