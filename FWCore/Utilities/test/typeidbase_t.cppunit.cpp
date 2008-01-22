/*----------------------------------------------------------------------

Test program for edm::TypeIDBase class.
Changed by Viji on 29-06-2005

$Id: typeidbase_t.cppunit.cpp,v 1.3 2008/01/22 20:41:14 muzaffar Exp $
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/TypeIDBase.h"

class testTypeIDBase: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testTypeIDBase);

CPPUNIT_TEST(equalityTest);
CPPUNIT_TEST(copyTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void equalityTest();
  void copyTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTypeIDBase);

namespace edmtest {
  struct empty { };
}

void testTypeIDBase::equalityTest()

{
  edmtest::empty e;
  edm::TypeIDBase id1(typeid(e));
  edm::TypeIDBase id2(typeid(e));
  
  CPPUNIT_ASSERT(!(id1 < id2));
  CPPUNIT_ASSERT(!(id2 < id1));

  std::string n1(id1.name());
  std::string n2(id2.name());

  CPPUNIT_ASSERT(n1==n2);
}

void testTypeIDBase::copyTest()
{
  edmtest::empty e;
  edm::TypeIDBase id1(typeid(e));

  edm::TypeIDBase id3=id1;
  CPPUNIT_ASSERT(!(id1 < id3));
  CPPUNIT_ASSERT(!(id3 < id1));

  std::string n1(id1.name());
  std::string n3(id3.name());
  CPPUNIT_ASSERT(n1== n3);
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
