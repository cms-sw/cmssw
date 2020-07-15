#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include <cmath>

class testPolynomial : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPolynomial);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPolynomial);

void testPolynomial::checkAll() {
  double c[] = {1, 2, 3, 4, 5};
  double x = 10;
  const double epsilon = 1.e-5;
  funct::Polynomial<0> poly0(c);
  CPPUNIT_ASSERT(fabs(poly0(x) - c[0]) < epsilon);
  funct::Polynomial<1> poly1(c);
  CPPUNIT_ASSERT(fabs(poly1(x) - (c[0] + x * c[1])) < epsilon);
  funct::Polynomial<2> poly2(c);
  CPPUNIT_ASSERT(fabs(poly2(x) - (c[0] + x * c[1] + x * x * c[2])) < epsilon);
  funct::Polynomial<3> poly3(c);
  CPPUNIT_ASSERT(fabs(poly3(x) - (c[0] + x * c[1] + x * x * c[2] + x * x * x * c[3])) < epsilon);
}
