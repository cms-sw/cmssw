#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

#include <string>

struct MyDDSingleton : public cms::DDSingleton<std::string, MyDDSingleton> {};

class testDDSingleton : public CppUnit::TestFixture {
  
  CPPUNIT_TEST_SUITE( testDDSingleton );
  CPPUNIT_TEST( testEquality );
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp() override{}
  void tearDown() override {}
  void testEquality();

private:
  MyDDSingleton m_singleton1;  
  MyDDSingleton m_singleton2;
};

void
testDDSingleton::testEquality()
{
  *m_singleton1 = "\"Hello\"";
  std::cout << "\n"
	    << *m_singleton1 << " is the same as "
	    << *m_singleton2 << "\n";
  CPPUNIT_ASSERT( *m_singleton1 == *m_singleton2 );

  *m_singleton2 = "\"World!\"";
  std::cout << "\n"
	    << *m_singleton1 << " is the same as "
	    << *m_singleton2 << "\n";
  CPPUNIT_ASSERT( *m_singleton1 == *m_singleton2 );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testDDSingleton );
