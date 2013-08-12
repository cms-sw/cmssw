/*----------------------------------------------------------------------

Test program for edm::TypeIDBase class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"

class testSTDNamespaceAdder: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testSTDNamespaceAdder);

CPPUNIT_TEST(tests);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void tests();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSTDNamespaceAdder);

void testSTDNamespaceAdder::tests()
{
   using edm::root::stdNamespaceAdder;

   static const std::string kVectorBlahNoNS("vector<Blah>");
   static const std::string kVectorBlahNS("std::vector<Blah>");
   static const std::string kTVectorBlah("Tvector<Blah>");
   static const std::string kVectorVectorBlahNoNS("vector<vector<Blah> >");
   static const std::string kVectorVectorBlahNS("std::vector<std::vector<Blah> >");

   //cout <<" substitution \""<<kVectorVectorBlahNoNS<<"\" \""
   //     <<stdNamespaceAdder(kVectorVectorBlahNoNS)<<"\""<< std::endl;

   CPPUNIT_ASSERT(stdNamespaceAdder(kVectorBlahNoNS) == kVectorBlahNS);
   CPPUNIT_ASSERT(stdNamespaceAdder(kVectorBlahNS) == kVectorBlahNS);
   CPPUNIT_ASSERT(stdNamespaceAdder(kTVectorBlah) == kTVectorBlah);
   CPPUNIT_ASSERT(stdNamespaceAdder(kVectorVectorBlahNoNS) == kVectorVectorBlahNS);
   CPPUNIT_ASSERT(stdNamespaceAdder(kVectorVectorBlahNS) == kVectorVectorBlahNS);
   
}
//#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
