// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     pluginfactory_t
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
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edmplugintest {
  struct DummyBase {};

  struct Dummy : public DummyBase {};
}  // namespace edmplugintest

typedef edmplugin::PluginFactory<edmplugintest::DummyBase*(void)> FactoryType;
EDM_REGISTER_PLUGINFACTORY(FactoryType, "Test Dummy");

DEFINE_EDM_PLUGIN(FactoryType, edmplugintest::Dummy, "Dummy");

TEST_CASE("PluginFactory", "[PluginManager]") {
  static bool alreadySetup = false;
  if (!alreadySetup) {
    alreadySetup = true;
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

  SECTION("test") {
    using namespace edmplugin;

    std::unique_ptr<edmplugintest::DummyBase> p(FactoryType::get()->create("Dummy"));
    REQUIRE(nullptr != p.get());
  }

  SECTION("tryToCreate") {
    using namespace edmplugin;
    REQUIRE(nullptr == FactoryType::get()->tryToCreate("ThisDoesNotExist").get());

    std::unique_ptr<edmplugintest::DummyBase> p(FactoryType::get()->tryToCreate("Dummy"));
    REQUIRE(nullptr != p.get());
  }
}
