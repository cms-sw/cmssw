#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Integral.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <iostream>
class testIntegral : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testIntegral);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

NUMERICAL_INTEGRAL(X, Expression, 10000);

struct gauss {
  static const double c;
  double operator()(double x) const {
    return c * exp(- x * x);
  }
};

struct gauss2 : public gauss { };

struct gaussPrimitive {
  double operator()(double x) const { return erf(x); }
};

const double gauss::c = 2./sqrt(M_PI);

NUMERICAL_FUNCT_INTEGRAL(gauss, 10000);

DECLARE_FUNCT_PRIMITIVE(gauss2, gaussPrimitive);

CPPUNIT_TEST_SUITE_REGISTRATION(testIntegral);

void testIntegral::checkAll() {
  using namespace funct;
  X x; 
  double epsilon = 1.e-6;
  // symbolic primitive exists
  CPPUNIT_ASSERT(fabs(integral<X>(x, 0, 1) - 0.5) < epsilon);
  CPPUNIT_ASSERT(fabs(integral<X>(x^num<2>(), 0, 1) - 1./3.) < epsilon);
  // numerical integration
  Expression f_x = x;
  CPPUNIT_ASSERT(fabs(integral<X>(f_x, 0, 1) - 0.5) < epsilon);

  Expression f_x2 = (x ^ num<2>());
  CPPUNIT_ASSERT(fabs(integral<X>(f_x2, 0, 1) - 1./3.) < epsilon);

  Parameter c("c", gauss::c);
  Expression f = c * exp(-(x^num<2>()));
  CPPUNIT_ASSERT(fabs(integral<X>(f, 0, 1) - erf(1)) < epsilon);

  // numerical integration
  gauss g;
  CPPUNIT_ASSERT(fabs(integral(g, 0, 1) - erf(1)) < epsilon);
  // user-defined integration
  gauss2 g2;
  CPPUNIT_ASSERT(fabs(integral(g2, 0, 1) - erf(1)) < epsilon);
  // automatic (trivial) integration
  Parameter pi("pi", M_PI);
  CPPUNIT_ASSERT(fabs(integral(pi, 0, 2) - 2*pi()) < epsilon);
}

