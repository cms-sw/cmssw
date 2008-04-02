#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/FunctionsIO.h"
#include "PhysicsTools/Utilities/interface/Variables.h"
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

void testFunctionsIO::checkAll() {
  using namespace funct;
  X x; Y y;
  {
    std::ostringstream str;
    str << x;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "x");
  }
  {
    std::ostringstream str;
    str << exp(x);
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "exp(x)");
  }
  {
    std::ostringstream str;
    str << x + y;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "x + y");
  }
  {
    std::ostringstream str;
    str << x * y;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "x y");
  }
  {
    std::ostringstream str;
    str << x / y;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "x/y");
  }
  {
    std::ostringstream str;
    str << - x;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "-x");
  }
  {
    std::ostringstream str;
    str << x - y;
    std::cerr << str.str() << std::endl;
    CPPUNIT_ASSERT(str.str() == "x - y");
  }
}
