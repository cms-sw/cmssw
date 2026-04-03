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
#include "catch2/catch_all.hpp"
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/CacheParser.h"
class TestCacheParser {
public:
  static std::string replaceSpaces(std::string& iString) { return edmplugin::CacheParser::replaceSpaces(iString); }
  static std::string restoreSpaces(std::string& iString) { return edmplugin::CacheParser::restoreSpaces(iString); }
  static void write(const std::map<std::string, std::vector<edmplugin::PluginInfo> >& iIn, std::ostream& oOut) {
    edmplugin::CacheParser::write(iIn, oOut);
  }
  static void read(std::istream& iIn,
                   const std::filesystem::path& iDirectory,
                   std::map<std::string, std::vector<edmplugin::PluginInfo> >& oOut) {
    edmplugin::CacheParser::read(iIn, iDirectory, oOut);
  }
};

TEST_CASE("CacheParser", "[PluginManager]") {
  SECTION("space") {
    using namespace edmplugin;

    const std::string kNoSpace("abcDefla");
    std::string unchanged(kNoSpace);

    REQUIRE(kNoSpace == TestCacheParser::replaceSpaces(unchanged));
    REQUIRE(kNoSpace == unchanged);
    REQUIRE(kNoSpace == TestCacheParser::restoreSpaces(unchanged));
    REQUIRE(kNoSpace == unchanged);

    const std::string kWithSpace("abc Def");
    std::string changed(kWithSpace);

    const std::string kSpaceReplaced("abc%Def");
    REQUIRE(kSpaceReplaced == TestCacheParser::replaceSpaces(changed));
    REQUIRE(kSpaceReplaced == changed);

    REQUIRE(kWithSpace == TestCacheParser::restoreSpaces(changed));
    REQUIRE(kWithSpace == changed);
  }

  SECTION("readWrite") {
    using namespace edmplugin;
    std::map<std::string, std::vector<PluginInfo> > categoryToInfos;
    PluginInfo info;
    info.loadable_ = "pluginA.so";
    info.name_ = "AlphaClass";
    categoryToInfos["Cat One"].push_back(info);
    info.name_ = "BetaClass<Itl >";
    categoryToInfos["Cat Two"].push_back(info);

    const std::string match("pluginA.so AlphaClass Cat%One\npluginA.so BetaClass<Itl%> Cat%Two\n");
    {
      std::stringstream ss;
      TestCacheParser::write(categoryToInfos, ss);
      REQUIRE(match == ss.str());
    }

    //check that the path is removed
    categoryToInfos.clear();

    info.loadable_ = "/enee/menee/minee/mo/pluginA.so";
    info.name_ = "AlphaClass";
    categoryToInfos["Cat One"].push_back(info);
    info.name_ = "BetaClass<Itl >";
    categoryToInfos["Cat Two"].push_back(info);

    {
      std::stringstream ss;
      TestCacheParser::write(categoryToInfos, ss);
      REQUIRE(match == ss.str());

      std::map<std::string, std::vector<PluginInfo> > readValues;
      TestCacheParser::read(ss, "/enee/menee/minee/mo", readValues);
      REQUIRE(categoryToInfos.size() == readValues.size());

      {
        std::stringstream ssFinalWrite;
        TestCacheParser::write(readValues, ssFinalWrite);
        std::cout << ssFinalWrite.str() << std::endl;
        REQUIRE(match == ssFinalWrite.str());
      }
    }
  }
}
