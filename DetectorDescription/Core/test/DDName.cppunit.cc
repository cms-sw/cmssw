#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/Singleton.h"
#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

class testDDName : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDName);
  CPPUNIT_TEST(checkNames);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() override{}
  void tearDown() override {}
  void buildIt();
  void testloading();
  void checkNames();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDName);

void
testDDName::buildIt() {
  DDName name;
  DDName anotherName;
  std::cerr << "\nDDName name=" << name << " id=" << name.id()
	    << ", another name=" << anotherName << " id=" << anotherName.id();
  int a[100000]{};
  for( int i : a )
  {
    DDName myName( std::to_string( i ));
  }
  DDName::Registry & reg = DDI::Singleton<DDName::Registry>::instance();
  DDName::Registry::size_type sz = reg.size();
  std::cerr << "\nTotal DDNames: " << sz;
}

void
testDDName::testloading() {
  DDName name;
  DDName anotherName;
  std::ostringstream  os;
  os << "DDName name=" << name << " id=" << name.id()
     << ", another name=" << anotherName << " id=" << anotherName.id();
  std::string str( "DDName name=anonymous:anonymous id=0, another name=anonymous:anonymous id=0" );
  if( os.str() != str ) std::cerr << "not the same!" << std::endl;
  CPPUNIT_ASSERT( os.str() == str );
}

void
testDDName::checkNames() {
  buildIt();
  testloading();
}
