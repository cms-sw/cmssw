#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Simplify.h"
#include <sstream>
#include <iostream>
class testFunctionsIO : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testFunctionsIO);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testFunctionsIO);

#define CHECK(EXPR, STRING) \
{ \
  std::ostringstream str; \
  str << EXPR; \
  std::cerr << str.str() << std::endl; \
  CPPUNIT_ASSERT(str.str() == STRING); \
} \
 \
struct __useless_igonreme

void testFunctionsIO::checkAll() {
  using namespace funct;
  X x; Y y;
  CHECK(x, "x");
  CHECK(-x, "-x");
  CHECK(exp(x), "exp(x)");
  CHECK(x + y, "x + y");
  CHECK(x - y, "x - y");
  CHECK(x * y, "x y");
  CHECK(x / y, "x/y");
  CHECK(num<1>(), "1");
  CHECK(num<2>(), "2");
  CHECK(num<3>(), "3");
  CHECK(num<2>()+num<3>(), "5"); 
  CHECK(num<2>()-num<3>(), "-1"); 
  CHECK(num<2>()*num<3>(), "6"); 
  CHECK(num<2>()/num<3>(), "2/3"); 
}
