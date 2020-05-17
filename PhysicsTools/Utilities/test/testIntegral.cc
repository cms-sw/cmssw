#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Integral.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"

class testIntegral : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testIntegral);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

NUMERICAL_INTEGRAL(X, Expression, TrapezoidIntegrator);

struct gauss {
  static const double c;
  double operator()(double x) const { return c * exp(-x * x); }
};

struct gauss2 : public gauss {};

struct gauss3 : public gauss {};

struct gauss4 : public gauss {};

struct gaussPrimitive {
  double operator()(double x) const { return erf(x); }
};

const double gauss::c = 2. / sqrt(M_PI);

NUMERICAL_FUNCT_INTEGRAL(gauss, TrapezoidIntegrator);

DECLARE_FUNCT_PRIMITIVE(gauss2, gaussPrimitive);

NUMERICAL_FUNCT_INTEGRAL(gauss3, GaussLegendreIntegrator);

NUMERICAL_FUNCT_INTEGRAL(gauss4, GaussIntegrator);

CPPUNIT_TEST_SUITE_REGISTRATION(testIntegral);

void testIntegral::checkAll() {
  using namespace funct;
  X x;
  double epsilon = 1.e-6;
  // symbolic primitive exists
  CPPUNIT_ASSERT(fabs(integral<X>(x, 0, 1) - 0.5) < epsilon);
  CPPUNIT_ASSERT(fabs(integral<X>(x ^ num<2>(), 0, 1) - 1. / 3.) < epsilon);

  TrapezoidIntegrator integrator(1000);

  // numerical integration
  Expression f_x = x;
  CPPUNIT_ASSERT(fabs(integral<X>(f_x, 0, 1, integrator) - 0.5) < epsilon);

  Expression f_x2 = (x ^ num<2>());
  CPPUNIT_ASSERT(fabs(integral<X>(f_x2, 0, 1, integrator) - 1. / 3.) < epsilon);

  Parameter c("c", gauss::c);
  Expression f = c * exp(-(x ^ num<2>()));
  CPPUNIT_ASSERT(fabs(integral<X>(f, 0, 1, integrator) - erf(1)) < epsilon);

  // trapezoid integration
  gauss g;
  CPPUNIT_ASSERT(fabs(integral_f(g, 0, 1, integrator) - erf(1)) < epsilon);

  // user-defined integration
  gauss2 g2;
  CPPUNIT_ASSERT(fabs(integral_f(g2, 0, 1) - erf(1)) < epsilon);

  // Gauss-Legendre integration
  gauss3 g3;
  GaussLegendreIntegrator integrator2(10000, 1.e-5);
  CPPUNIT_ASSERT(fabs(integral_f(g3, 0, 1, integrator2) - erf(1)) < epsilon);

  // Gauss-Legendre integration
  gauss4 g4;
  GaussIntegrator integrator3(1.e-6);
  CPPUNIT_ASSERT(fabs(integral_f(g4, 0, 1, integrator3) - erf(1)) < epsilon);

  // automatic (trivial) integration
  Parameter pi("pi", M_PI);
  CPPUNIT_ASSERT(fabs(integral_f(pi, 0, 2) - 2 * pi()) < epsilon);
}
