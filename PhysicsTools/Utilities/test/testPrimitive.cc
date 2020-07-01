#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Primitive.h"
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Simplify.h"
#include <sstream>
#include <iostream>
class testPrimitives : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPrimitives);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPrimitives);

#define CHECK(EXPR, STRING)                                \
  {                                                        \
    std::ostringstream str;                                \
    str << EXPR;                                           \
    std::cerr << #EXPR << " = " << str.str() << std::endl; \
    CPPUNIT_ASSERT(str.str() == STRING);                   \
  }                                                        \
                                                           \
  struct __useless_igonreme

void testPrimitives::checkAll() {
  using namespace funct;
  X x;
  Y y;
  Z z;
  x = 1, y = 2, z = 3;
  funct::Numerical<1> _1;
  funct::Numerical<2> _2;
  funct::Numerical<3> _3;
  funct::Numerical<4> _4;
  funct::Numerical<-1> _m1;
  funct::Numerical<-2> _m2;
  funct::Numerical<-3> _m3;
  funct::Numerical<-4> _m4;

  CHECK(primitive<X>(x), "x^2/2");
  CHECK(primitive<X>(x ^ _2), "x^3/3");
  CHECK(primitive<X>(x ^ _3), "x^4/4");
  CHECK(primitive<X>(x ^ _4), "x^5/5");
  CHECK(primitive<X>(x ^ _m4), "-( 1/3 )/x^3");
  CHECK(primitive<X>(_1 / (x ^ _4)), "-( 1/3 )/x^3");
  CHECK(primitive<X>((_1 / x) ^ _4), "-( 1/3 )/x^3");
  CHECK(primitive<X>(x ^ (_3 / _4)), "4/7 x^( 7/4 )");
  CHECK(primitive<X>(sqrt(x)), "2/3 x^( 3/2 )");
  CHECK(primitive<X>(exp(x)), "exp(x)");
  CHECK(primitive<X>(log(x)), "x ( log(x) - 1 )");
  CHECK(primitive<X>(sgn(x)), "abs(x)");
  CHECK(primitive<X>(sin(x)), "-cos(x)");
  CHECK(primitive<X>(cos(x)), "sin(x)");
  CHECK(primitive<X>(tan(x)), "-log(abs(cos(x)))");
  CHECK(primitive<X>(_1 / x), "log(abs(x))");
  CHECK((primitive<X>(x ^ _m1)), "log(abs(x))");
  CHECK((primitive<X>(_1 / (cos(x) ^ _2))), "tan(x)");
  CHECK((primitive<X>(cos(x) ^ _m2)), "tan(x)");
  CHECK((primitive<X>(_1 / (sin(x) ^ _2))), "-1/tan(x)");
  CHECK((primitive<X>(sin(x) ^ _m2)), "-1/tan(x)");
  CHECK(primitive<X>(x * exp(x)), "exp(x) ( x - 1 )");
  CHECK(primitive<X>(exp(x) * x), "exp(x) ( x - 1 )");
}
