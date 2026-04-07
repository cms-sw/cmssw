// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginmanager_t
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

// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/PluginManager/test/DummyFactory.h"

class DummyTestPlugin : public edmplugin::PluginFactoryBase {
public:
  DummyTestPlugin(const std::string& iName) : name_(iName) { finishedConstruction(); }
  const std::string& category() const { return name_; }
  std::vector<edmplugin::PluginInfo> available() const { return std::vector<edmplugin::PluginInfo>(); }
  const std::string name_;
};
/*
struct Catcher {
  std::string lastSeen_;
  
  void catchIt(const edmplugin::PluginFactoryBase* iFactory) {
    lastSeen_=iFactory->category();
  }
};
*/

namespace testedmplugin {
  struct DummyThree : public DummyBase {
    int value() const { return 3; }
  };
}  // namespace testedmplugin

DEFINE_EDM_PLUGIN(testedmplugin::DummyFactory, testedmplugin::DummyThree, "DummyThree");

TEST_CASE("PluginManager", "[PluginManager]") {
  SECTION("test") {
    using namespace edmplugin;
    using namespace testedmplugin;
    REQUIRE_THROWS_AS(PluginManager::get(), cms::Exception);

    PluginManager::Config config;
    REQUIRE_THROWS_AS(PluginManager::configure(config), cms::Exception);

    edmplugin::PluginManager& db = edmplugin::PluginManager::configure(edmplugin::standard::config());

    std::string toLoadCategory("Test Dummy");
    std::string toLoadPlugin;
    unsigned int nTimesAsked = 0;

    edmplugin::PluginManager::get()->askedToLoadCategoryWithPlugin_.connect(
        [&](std::string const& iCategory, std::string const& iPlugin) {
          //std::cout <<iCategory<<" "<<iPlugin<<std::endl;
          REQUIRE(toLoadCategory == iCategory);
          REQUIRE(toLoadPlugin == iPlugin);
          ++nTimesAsked;
        });

    unsigned int nTimesGoingToLoad = 0;
    edmplugin::PluginManager::get()->goingToLoad_.connect(
        [&nTimesGoingToLoad](const std::filesystem::path&) { ++nTimesGoingToLoad; });

    unsigned int nTimesLoaded = 0;
    edmplugin::PluginManager::get()->justLoaded_.connect(
        [&nTimesLoaded](const edmplugin::SharedLibrary&) { ++nTimesLoaded; });

    toLoadPlugin = "DummyOne";
    std::unique_ptr<DummyBase> ptr(DummyFactory::get()->create("DummyOne"));
    REQUIRE(1 == ptr->value());
    REQUIRE(nTimesAsked == 1);
    REQUIRE(nTimesGoingToLoad == 1);
    REQUIRE(nTimesLoaded == 1);
    REQUIRE(db.loadableFor("Test Dummy", "DummyThree") == "static");
    std::unique_ptr<DummyBase> ptr2(DummyFactory::get()->create("DummyThree"));
    REQUIRE(3 == ptr2->value());
    REQUIRE(nTimesAsked == 1);  //no request to load
    REQUIRE(nTimesGoingToLoad == 1);
    REQUIRE(nTimesLoaded == 1);

    toLoadPlugin = "DoesNotExist";
    REQUIRE_THROWS_AS(DummyFactory::get()->create("DoesNotExist"), cms::Exception);
    REQUIRE(nTimesAsked == 2);  //request happens even though it failed
    REQUIRE(nTimesGoingToLoad == 1);
    REQUIRE(nTimesLoaded == 1);
  }
}
