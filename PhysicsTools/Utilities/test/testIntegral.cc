#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Integral.h"
#include "PhysicsTools/Utilities/interface/Variables.h"

class testIntegral : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testIntegral);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testIntegral);

void testIntegral::checkAll() {
  using namespace funct;
  X x; 
  double epsilon = 1.e-6;
  CPPUNIT_ASSERT(fabs(integral<X>(x, 0, 1) - 0.5) < epsilon);
  CPPUNIT_ASSERT(fabs(integral<X>(x^num<2>(), 0, 1) - 1./3.) < epsilon);
}

