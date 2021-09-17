#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

TEST_CASE("Test TriggerNames", "[TriggerNames]") {
  edm::ParameterSet pset;
  const std::vector<std::string> names = {{"b1"}, {"b2"}, {"a1"}, {"z5"}};
  pset.addParameter<std::vector<std::string>>("@trigger_paths", names);
  pset.registerIt();

  SECTION("Default constructed") {
    edm::TriggerNames defaultConst;

    REQUIRE(defaultConst.size() == 0);
    REQUIRE(defaultConst.triggerNames().empty());
    REQUIRE(defaultConst.triggerIndex("not here") == defaultConst.size());
    REQUIRE_THROWS_AS(defaultConst.triggerName(0), std::exception);
  }

  SECTION("No names") {
    edm::ParameterSet pset;
    const std::vector<std::string> names;
    pset.addParameter<std::vector<std::string>>("@trigger_paths", names);
    pset.registerIt();

    edm::TriggerNames noNames(pset);

    REQUIRE(noNames.size() == 0);
    REQUIRE(noNames.triggerNames().empty());
    REQUIRE(noNames.parameterSetID() == pset.id());
    REQUIRE(noNames.triggerIndex("not here") == noNames.size());
    REQUIRE_THROWS_AS(noNames.triggerName(0), std::exception);
  }

  SECTION("multiple names") {
    edm::TriggerNames tNames(pset);

    REQUIRE(tNames.size() == names.size());
    REQUIRE(tNames.triggerNames() == names);
    REQUIRE(tNames.parameterSetID() == pset.id());
    REQUIRE(tNames.triggerIndex("not here") == tNames.size());
    REQUIRE(tNames.triggerIndex("b1") == 0);
    REQUIRE(tNames.triggerIndex("b2") == 1);
    REQUIRE(tNames.triggerIndex("a1") == 2);
    REQUIRE(tNames.triggerIndex("z5") == 3);
    REQUIRE(tNames.triggerName(0) == "b1");
    REQUIRE(tNames.triggerName(1) == "b2");
    REQUIRE(tNames.triggerName(2) == "a1");
    REQUIRE(tNames.triggerName(3) == "z5");
    REQUIRE_THROWS_AS(tNames.triggerName(names.size()), std::exception);
  }

  SECTION("copy constructor") {
    auto temp = std::make_unique<edm::TriggerNames>(pset);

    edm::TriggerNames tNames(*temp);
    temp.release();

    REQUIRE(tNames.size() == names.size());
    REQUIRE(tNames.triggerNames() == names);
    REQUIRE(tNames.parameterSetID() == pset.id());
    REQUIRE(tNames.triggerIndex("not here") == tNames.size());
    REQUIRE(tNames.triggerIndex("b1") == 0);
    REQUIRE(tNames.triggerIndex("b2") == 1);
    REQUIRE(tNames.triggerIndex("a1") == 2);
    REQUIRE(tNames.triggerIndex("z5") == 3);
    REQUIRE(tNames.triggerName(0) == "b1");
    REQUIRE(tNames.triggerName(1) == "b2");
    REQUIRE(tNames.triggerName(2) == "a1");
    REQUIRE(tNames.triggerName(3) == "z5");
    REQUIRE_THROWS_AS(tNames.triggerName(names.size()), std::exception);
  }

  SECTION("move constructor") {
    auto temp = std::make_unique<edm::TriggerNames>(pset);

    edm::TriggerNames tNames(std::move(*temp));
    temp.release();

    REQUIRE(tNames.size() == names.size());
    REQUIRE(tNames.triggerNames() == names);
    REQUIRE(tNames.parameterSetID() == pset.id());
    REQUIRE(tNames.triggerIndex("not here") == tNames.size());
    REQUIRE(tNames.triggerIndex("b1") == 0);
    REQUIRE(tNames.triggerIndex("b2") == 1);
    REQUIRE(tNames.triggerIndex("a1") == 2);
    REQUIRE(tNames.triggerIndex("z5") == 3);
    REQUIRE(tNames.triggerName(0) == "b1");
    REQUIRE(tNames.triggerName(1) == "b2");
    REQUIRE(tNames.triggerName(2) == "a1");
    REQUIRE(tNames.triggerName(3) == "z5");
    REQUIRE_THROWS_AS(tNames.triggerName(names.size()), std::exception);
  }

  SECTION("operator=") {
    auto temp = std::make_unique<edm::TriggerNames>(pset);

    edm::TriggerNames tNames;
    tNames = (*temp);
    temp.release();

    REQUIRE(tNames.size() == names.size());
    REQUIRE(tNames.triggerNames() == names);
    REQUIRE(tNames.parameterSetID() == pset.id());
    REQUIRE(tNames.triggerIndex("not here") == tNames.size());
    REQUIRE(tNames.triggerIndex("b1") == 0);
    REQUIRE(tNames.triggerIndex("b2") == 1);
    REQUIRE(tNames.triggerIndex("a1") == 2);
    REQUIRE(tNames.triggerIndex("z5") == 3);
    REQUIRE(tNames.triggerName(0) == "b1");
    REQUIRE(tNames.triggerName(1) == "b2");
    REQUIRE(tNames.triggerName(2) == "a1");
    REQUIRE(tNames.triggerName(3) == "z5");
    REQUIRE_THROWS_AS(tNames.triggerName(names.size()), std::exception);
  }

  SECTION("operator= with move") {
    auto temp = std::make_unique<edm::TriggerNames>(pset);

    edm::TriggerNames tNames;
    tNames = std::move(*temp);
    temp.release();

    REQUIRE(tNames.size() == names.size());
    REQUIRE(tNames.triggerNames() == names);
    REQUIRE(tNames.parameterSetID() == pset.id());
    REQUIRE(tNames.triggerIndex("not here") == tNames.size());
    REQUIRE(tNames.triggerIndex("b1") == 0);
    REQUIRE(tNames.triggerIndex("b2") == 1);
    REQUIRE(tNames.triggerIndex("a1") == 2);
    REQUIRE(tNames.triggerIndex("z5") == 3);
    REQUIRE(tNames.triggerName(0) == "b1");
    REQUIRE(tNames.triggerName(1) == "b2");
    REQUIRE(tNames.triggerName(2) == "a1");
    REQUIRE(tNames.triggerName(3) == "z5");
    REQUIRE_THROWS_AS(tNames.triggerName(names.size()), std::exception);
  }
}
