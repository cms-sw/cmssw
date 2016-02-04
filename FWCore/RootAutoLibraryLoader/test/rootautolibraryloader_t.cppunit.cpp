/*----------------------------------------------------------------------

Test program for RootAutoLibraryLoader class.
Changed by Viji on 29-06-2005

$Id: rootautolibraryloader_t.cppunit.cpp,v 1.1 2010/09/03 19:04:06 chrjones Exp $
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>

#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#define private public
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#undef private

class testRootAutoLibraryLoader: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testRootAutoLibraryLoader);

CPPUNIT_TEST(testString);

CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void testString();
};

static edm::RootAutoLibraryLoader* createLoader()
{
   static edm::RootAutoLibraryLoader s_loader;
   return &s_loader;
}
///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRootAutoLibraryLoader);

void testRootAutoLibraryLoader::testString()
{
   if(not edmplugin::PluginManager::isAvailable()) {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
   }
   edm::RootAutoLibraryLoader* loader = createLoader();
   
   CPPUNIT_ASSERT(0!=loader->GetClass("edm::Wrapper<std::basic_string<char> >", true));
   
   CPPUNIT_ASSERT(0==loader->GetClass("ThisClassDoesNotExist",true));
}
