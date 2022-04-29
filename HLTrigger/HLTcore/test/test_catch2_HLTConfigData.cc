#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace {
  edm::ParameterSet buildModulePSet(std::string const& iLabel, std::string const& iType, std::string const& iEDMType) {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@module_label", iLabel);
    pset.addParameter<std::string>("@module_type", iType);
    pset.addParameter<std::string>("@module_edm_type", iEDMType);
    return pset;
  }
}  // namespace

TEST_CASE("Test HLTConfigData", "[HLTConfigData]") {
  SECTION("TriggerPaths") {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@process_name", "TEST");
    const std::vector<std::string> names = {{"b1"}, {"b2"}, {"a1"}, {"z5"}};
    {
      edm::ParameterSet tpset;
      tpset.addParameter<std::vector<std::string>>("@trigger_paths", names);
      pset.addParameter<edm::ParameterSet>("@trigger_paths", tpset);
    }
    pset.addParameter<std::vector<std::string>>("b1", {{"f1"}, {"f2"}});
    pset.addParameter<std::vector<std::string>>("b2", {{"f3"}, {"#"}, {"c1"}, {"c2"}, {"@"}});
    pset.addParameter<std::vector<std::string>>("a1", {{"f1"}, {"f4"}});
    pset.addParameter<std::vector<std::string>>("z5", {{"f5"}});

    pset.addParameter<edm::ParameterSet>("f1", buildModulePSet("f1", "F1Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f2", buildModulePSet("f2", "F2Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f3", buildModulePSet("f3", "F3Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f4", buildModulePSet("f4", "F4Filter", "EDFilter"));
    pset.addParameter<edm::ParameterSet>("f5", buildModulePSet("f5", "F5Filter", "EDFilter"));

    pset.addParameter<edm::ParameterSet>("c1", buildModulePSet("c1", "CProducer", "EDProducer"));
    pset.addParameter<edm::ParameterSet>("c2", buildModulePSet("c2", "CProducer", "EDProducer"));
    pset.registerIt();

    HLTConfigData cd(&pset);

    SECTION("check paths") {
      REQUIRE(cd.size() == 4);
      REQUIRE(cd.triggerName(0) == names[0]);
      REQUIRE(cd.triggerName(1) == names[1]);
      REQUIRE(cd.triggerName(2) == names[2]);
      REQUIRE(cd.triggerName(3) == names[3]);
      //cd.dump("Triggers");
    }

    SECTION("check modules on paths") {
      REQUIRE(cd.size(0) == 2);
      REQUIRE(cd.moduleLabel(0, 0) == "f1");
      REQUIRE(cd.moduleLabel(0, 1) == "f2");

      REQUIRE(cd.size(1) == 3);
      REQUIRE(cd.moduleLabel(1, 0) == "f3");
      REQUIRE(cd.moduleLabel(1, 1) == "c1");
      REQUIRE(cd.moduleLabel(1, 2) == "c2");

      REQUIRE(cd.size(2) == 2);
      REQUIRE(cd.moduleLabel(2, 0) == "f1");
      REQUIRE(cd.moduleLabel(2, 1) == "f4");

      REQUIRE(cd.size(3) == 1);
      REQUIRE(cd.moduleLabel(3, 0) == "f5");
      //cd.dump("Modules");
    }
  }
}
