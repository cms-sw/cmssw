#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/NthDerivative.h"
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Simplify.h"
#include <sstream>
#include <iostream>
class testDerivatives : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDerivatives);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDerivatives);

#define CHECK(EXPR, STRING)                                \
  {                                                        \
    std::ostringstream str;                                \
    str << EXPR;                                           \
    std::cerr << #EXPR << " = " << str.str() << std::endl; \
    CPPUNIT_ASSERT(str.str() == STRING);                   \
  }                                                        \
                                                           \
  struct __useless_igonreme

void testDerivatives::checkAll() {
  using namespace funct;
  X x;
  Y y;
  Z z;
  x = 1, y = 2, z = 3;
  funct::Numerical<1> _1;
  funct::Numerical<2> _2;
  funct::Numerical<3> _3;

  CHECK(derivative<X>(_1), "0");
  CHECK(derivative<X>(x), "1");
  CHECK(derivative<Y>(x), "0");
  CHECK(derivative<X>(x ^ _2), "2 x");
  CHECK(derivative<X>(x ^ _3), "3 x^2");
  CHECK(derivative<X>(exp(x)), "exp(x)");
  CHECK(derivative<X>(log(x)), "1/x");
  CHECK(derivative<X>(sin(x)), "cos(x)");
  CHECK(derivative<X>(cos(x)), "-sin(x)");

  CHECK((nth_derivative<2, X>(sin(x))), "-sin(x)");
  CHECK((nth_derivative<3, X>(sin(x))), "-cos(x)");
  CHECK((nth_derivative<4, X>(sin(x))), "sin(x)");

  CHECK(derivative<X>(sin(x) * cos(x)), "cos(x)^2 - sin(x)^2");
  CHECK((nth_derivative<2, X>(sin(x) * cos(x))), "-4 sin(x) cos(x)");
  CHECK((nth_derivative<3, X>(sin(x) * cos(x))), "-4 ( cos(x)^2 - sin(x)^2 )");
  CHECK((nth_derivative<4, X>(sin(x) * cos(x))), "16 sin(x) cos(x)");
}
