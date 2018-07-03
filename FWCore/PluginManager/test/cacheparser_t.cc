// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     loadablemanager_t
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:38:29 EDT 2007
//

// system include files
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/CacheParser.h"

class TestCacheParser : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestCacheParser);
  CPPUNIT_TEST(testSpace);
  CPPUNIT_TEST(testReadWrite);
  CPPUNIT_TEST_SUITE_END();
public:
    void testSpace();
    void testReadWrite();
    void setUp() {}
    void tearDown() {}
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestCacheParser);

void
TestCacheParser::testSpace()
{
  using namespace edmplugin;

  const std::string kNoSpace("abcDefla");
  std::string unchanged(kNoSpace);
  
  CPPUNIT_ASSERT(kNoSpace == CacheParser::replaceSpaces(unchanged));
  CPPUNIT_ASSERT(kNoSpace == unchanged);
  CPPUNIT_ASSERT(kNoSpace == CacheParser::restoreSpaces(unchanged));
  CPPUNIT_ASSERT(kNoSpace == unchanged);
  
  const std::string kWithSpace("abc Def");
  std::string changed(kWithSpace);
  
  const std::string kSpaceReplaced("abc%Def");
  CPPUNIT_ASSERT(kSpaceReplaced == CacheParser::replaceSpaces(changed));
  CPPUNIT_ASSERT(kSpaceReplaced == changed);
  
  CPPUNIT_ASSERT(kWithSpace == CacheParser::restoreSpaces(changed));
  CPPUNIT_ASSERT(kWithSpace == changed);
}

void
TestCacheParser::testReadWrite()
{
  using namespace edmplugin;
  std::map<std::string, std::vector<PluginInfo> > categoryToInfos;
  PluginInfo info;
  info.loadable_="pluginA.so";
  info.name_="AlphaClass";
  categoryToInfos["Cat One"].push_back(info);
  info.name_="BetaClass<Itl >";
  categoryToInfos["Cat Two"].push_back(info);

  const std::string match("pluginA.so AlphaClass Cat%One\npluginA.so BetaClass<Itl%> Cat%Two\n");
  {
    std::stringstream s;
    CacheParser::write(categoryToInfos,s);
    CPPUNIT_ASSERT(match == s.str());
  }
  
  //check that the path is removed
  categoryToInfos.clear();
  
  info.loadable_="/enee/menee/minee/mo/pluginA.so";
  info.name_="AlphaClass";
  categoryToInfos["Cat One"].push_back(info);
  info.name_="BetaClass<Itl >";
  categoryToInfos["Cat Two"].push_back(info);

  {
    std::stringstream s;
    CacheParser::write(categoryToInfos,s);    
    CPPUNIT_ASSERT(match == s.str());

    std::map<std::string, std::vector<PluginInfo> > readValues; 
    CacheParser::read(s,"/enee/menee/minee/mo",readValues);
    CPPUNIT_ASSERT(categoryToInfos.size() == readValues.size());
    
    {
      std::stringstream s;
      CacheParser::write(readValues,s);
      std::cout <<s.str()<<std::endl;
      CPPUNIT_ASSERT(match == s.str());
    }
  }
}  
