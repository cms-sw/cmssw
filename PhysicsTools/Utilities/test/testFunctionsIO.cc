#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Simplify.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include <sstream>
#include <iostream>
class testFunctionsIO : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFunctionsIO);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testFunctionsIO);

#define CHECK(EXPR, STRING)                                \
  {                                                        \
    std::ostringstream str;                                \
    str << EXPR;                                           \
    std::cerr << #EXPR << " = " << str.str() << std::endl; \
    CPPUNIT_ASSERT(str.str() == STRING);                   \
  }                                                        \
                                                           \
  struct __useless_igonreme

void testFunctionsIO::checkAll() {
  using namespace funct;
  X x;
  Y y;
  Z z;
  x = 1, y = 2, z = 3;
  funct::Numerical<0> _0;
  funct::Numerical<1> _1;
  funct::Numerical<2> _2;
  funct::Numerical<3> _3;
  funct::Numerical<-1> _m1;
  funct::Numerical<-2> _m2;
  funct::Numerical<-3> _m3;
  CHECK(x, "x");
  CHECK(-x, "-x");

  CHECK(sqrt(x), "sqrt(x)");
  CHECK(exp(x), "exp(x)");
  CHECK(log(x), "log(x)");
  CHECK(sin(x), "sin(x)");
  CHECK(cos(x), "cos(x)");
  CHECK(abs(x), "abs(x)");
  CHECK(sgn(x), "sgn(x)");

  CHECK(x + y, "x + y");
  CHECK(x - y, "x - y");
  CHECK(x * y, "x y");
  CHECK((x ^ y), "x^y");
  CHECK(x / y, "x/y");

  CHECK(_1, "1");
  CHECK(_2, "2");
  CHECK(_3, "3");

  CHECK(_2 + _3, "5");
  CHECK(_2 - _3, "-1");
  CHECK(_2 * _3, "6");
  CHECK(_2 / _3, "2/3");
  CHECK((_2 ^ _3), "8");
  CHECK((_2 ^ _m3), "1/8");
  CHECK((_m2 ^ _3), "-8");
  CHECK((_m2 ^ _m3), "-1/8");

  CHECK((fract<1, 2>()), "1/2");
  CHECK((fract<4, 2>()), "2");
  CHECK((fract<1, -2>()), "-1/2");

  CHECK(-x - y, "- x - y");
  CHECK(x + _1, "x + 1");
  CHECK(x - _1, "x - 1");
  CHECK(-x + _1, "- x + 1");
  CHECK(-x - _1, "- x - 1");

  CHECK(-(-x), "x");
  CHECK(-(x + y), "- x - y");

  // simplifications
  CHECK(x + x, "2 x");
  CHECK(_1 + x, "x + 1");
  CHECK(x + _0, "x");
  CHECK(_0 + x, "x");
  CHECK(_0 - x, "-x");
  CHECK(x - (-y), "x + y");
  CHECK(_3 * x + _2 * x, "5 x");

  CHECK(_0 * x, "0");
  CHECK(_1 * x, "x");
  CHECK(x * _2, "2 x");
  CHECK((_1 * fract<3, 4>()), "3/4");
  CHECK(_m1 * x, "-x");
  CHECK(x * (-y), "-x y");
  CHECK(_1 * (-x), "-x");
  CHECK(x * (y / z), "( x y )/z");
  CHECK((-x) * (-y), "x y");
  CHECK((x * y) * (-z), "-x y z");
  CHECK((-x) * y, "-x y");
  CHECK(_2 * (x / y), "( 2 x )/y");
  CHECK(x * _2, "2 x");
  CHECK((x ^ y) * (x ^ z), "x^( y + z )");

  CHECK(_0 / x, "0");
  CHECK(x / _1, "x");
  CHECK(x / _m1, "-x");
  CHECK(x / _m2, "-x/2");
  CHECK((-x) / y, "-x/y");
  CHECK(x / y / z, "x/( y z )");
  CHECK((_3 * x) / (_2 * y), "3/2 ( x/y )");

  CHECK((x ^ _1), "x");
  CHECK((x ^ _0), "1");
  CHECK((x ^ _m1), "1/x");
  CHECK((x ^ _m2), "1/x^2");
  CHECK((x ^ fract<1, 2>()), "sqrt(x)");
  CHECK(((x ^ y) ^ z), "x^( y + z )");

  CHECK(log(exp(x)), "x");
  CHECK(exp(log(x)), "x");
  CHECK((log(x ^ y)), "y log(x)");
  CHECK(exp(x) * exp(y), "exp(x + y)");
  CHECK(log(x * y), "log(x) + log(y)");
  CHECK(log(x / y), "log(x) - log(y)");

  CHECK(sin(-x), "-sin(x)");
  CHECK(cos(-x), "cos(x)");
  CHECK(sin(x) / cos(x), "tan(x)");
  CHECK(cos(x) * tan(x), "sin(x)");
  CHECK(sin(x) / tan(x), "cos(x)");
  CHECK((sin(x) ^ _2) + (cos(x) ^ _2), "1");

  CHECK(x * y + x * z, "x ( y + z )");

  Expression expr = sin(x) * cos(x);
  CHECK(expr, "sin(x) cos(x)");
}
