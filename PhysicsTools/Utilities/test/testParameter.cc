#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <cmath>

class testParameter : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testParameter);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testParameter);

void testParameter::checkAll() {
  double aVal = 123;
  std::string aName = "a";
  funct::Parameter a(aName, aVal);
  CPPUNIT_ASSERT(a.name() == aName);
  CPPUNIT_ASSERT(a.value() == aVal);
  double av = a;
  CPPUNIT_ASSERT(av == aVal);
  std::shared_ptr<double> ap = a;
  CPPUNIT_ASSERT(*ap == aVal);
  aVal = 234;
  a = aVal;
  CPPUNIT_ASSERT(a.value() == aVal);
  funct::Parameter a1 = a;
  CPPUNIT_ASSERT(a.value() == a1.value());
  CPPUNIT_ASSERT(a.name() == a1.name());
  a = 567;
  CPPUNIT_ASSERT(a.value() == a1.value());
  a1 = 123;
  CPPUNIT_ASSERT(a.value() == a1.value());
}
