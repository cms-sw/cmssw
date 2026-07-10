// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginfactorymanager_t
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:38:29 EDT 2007
//

// system include files
#include "catch2/catch_all.hpp"
#include <functional>
#include <iostream>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"

class DummyTestPlugin : public edmplugin::PluginFactoryBase {
public:
  DummyTestPlugin(const std::string& iName) : name_(iName) { finishedConstruction(); }
  const std::string& category() const { return name_; }
  std::vector<edmplugin::PluginInfo> available() const { return std::vector<edmplugin::PluginInfo>(); }
  const std::string name_;
};

struct Catcher {
  std::string lastSeen_;

  void catchIt(const edmplugin::PluginFactoryBase* iFactory) { lastSeen_ = iFactory->category(); }
};

TEST_CASE("PluginFactoryManager", "[PluginManager]") {
  SECTION("test") {
    using namespace edmplugin;
    using std::placeholders::_1;
    PluginFactoryManager& pfm = *(PluginFactoryManager::get());
    REQUIRE(pfm.begin() == pfm.end());

    Catcher catcher;
    pfm.newFactory_.connect(std::bind(std::mem_fn(&Catcher::catchIt), &catcher, _1));

    DummyTestPlugin one("one");
    REQUIRE((pfm.begin() != pfm.end()));
    REQUIRE(catcher.lastSeen_ == std::string("one"));
  }
}
