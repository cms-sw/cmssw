#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Operations.h"

class testVariables : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testVariables);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

DEFINE_VARIABLE(W, "w");
IMPLEMENT_VARIABLE(W);

CPPUNIT_TEST_SUITE_REGISTRATION(testVariables);

void testVariables::checkAll() {
  double value = 1.234, value1 = 2.345;
  funct::X x;
  x = value;
  double v = x;
  CPPUNIT_ASSERT(v == value);
  CPPUNIT_ASSERT(x() == v);
  x.set(value1);
  v = x;
  CPPUNIT_ASSERT(v == value1);
  CPPUNIT_ASSERT(x() == v);

  funct::W w;
  w = value;
  v = w;
  CPPUNIT_ASSERT(v == value);
  CPPUNIT_ASSERT(w() == v);

  funct::Y y;

  x = value;
  y = value1;
  double z;
  z = x + y;
  CPPUNIT_ASSERT(z = value + value1);
  z = x - y;
  CPPUNIT_ASSERT(z = value - value1);
  z = x * y;
  CPPUNIT_ASSERT(z = value * value1);
  z = x / y;
  CPPUNIT_ASSERT(z = value / value1);
  z = x ^ y;
  CPPUNIT_ASSERT(z = pow(value, value1));
  z = -x;
  CPPUNIT_ASSERT(z = -value);
}
