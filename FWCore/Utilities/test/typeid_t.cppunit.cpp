/*----------------------------------------------------------------------

Test program for edm::TypeID class.
Changed by Viji on 29-06-2005

$Id: typeid_t.cppunit.cpp,v 1.1 2007/03/04 04:40:20 wmtan Exp $
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/TypeID.h"

class testTypeid: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testTypeid);

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
CPPUNIT_TEST_SUITE_REGISTRATION(testTypeid);

namespace edmtest {
  struct empty { };
}

using std::cerr;
using std::endl;

void testTypeid::equalityTest()

{
  edmtest::empty e;
  edm::TypeID id1(e);
  edm::TypeID id2(e);
  
  CPPUNIT_ASSERT(!(id1 < id2));
  CPPUNIT_ASSERT(!(id2 < id1));

  std::string n1(id1.name());
  std::string n2(id2.name());

  CPPUNIT_ASSERT(n1==n2);
}

void testTypeid::copyTest()
{
  edmtest::empty e;
  edm::TypeID id1(e);

  edm::TypeID id3=id1;
  CPPUNIT_ASSERT(!(id1 < id3));
  CPPUNIT_ASSERT(!(id3 < id1));

  std::string n1(id1.name());
  std::string n3(id3.name());
  CPPUNIT_ASSERT(n1== n3);
}
