/*
 *  eventsetupscontroller_t.cc
 */

#include "catch2/catch_all.hpp"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>
#include <vector>

namespace {
  edm::ActivityRegistry activityRegistry;
}

TEST_CASE("EventSetupsController", "[Framework][EventSetup]") {
  SECTION("constructorTest") {
    edm::eventsetup::EventSetupsController esController;

    REQUIRE(esController.mustFinishConfiguration() == true);

    edm::ParameterSet pset;
    std::vector<std::string> emptyVStrings;
    pset.addParameter<std::vector<std::string> >("@all_esprefers", emptyVStrings);
    pset.addParameter<std::vector<std::string> >("@all_essources", emptyVStrings);
    pset.addParameter<std::vector<std::string> >("@all_esmodules", emptyVStrings);

    esController.makeProvider(pset, &activityRegistry);
  }
}
