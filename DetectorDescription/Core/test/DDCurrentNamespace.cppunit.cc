#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

class testDDCurrentNamespace : public CppUnit::TestFixture {
  
  CPPUNIT_TEST_SUITE( testDDCurrentNamespace );
  CPPUNIT_TEST( checkNamespace );
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp() override{}
  void tearDown() override {}
  void checkNamespace();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDCurrentNamespace);

void testDDCurrentNamespace::checkNamespace()
{
  std::cout << "\nDefault namespace is GLOBAL.\n";
  {
    DDCurrentNamespace ns1;
    std::cout << *ns1 << "\n";
    std::cout << DDCurrentNamespace() << "\n";
    CPPUNIT_ASSERT( DDCurrentNamespace()->c_str() == std::string{"GLOBAL"});
  }
  {
    std::cout << DDCurrentNamespace()->c_str() << "\n";
    CPPUNIT_ASSERT( DDCurrentNamespace()->c_str() == std::string{"GLOBAL"});
  }
  {
    DDCurrentNamespace ns;
    std::cout << *ns << "\n";
    CPPUNIT_ASSERT( *ns == std::string{"GLOBAL"});
  }
  {
    DDCurrentNamespace ns;
    *ns = "New namespace";
    std::cout << *ns << "\n";
    CPPUNIT_ASSERT( *ns == std::string{"New namespace"});
  }
  std::cout << DDCurrentNamespace() << "\n";
  CPPUNIT_ASSERT( DDCurrentNamespace()->c_str() == std::string{"New namespace"});
}
