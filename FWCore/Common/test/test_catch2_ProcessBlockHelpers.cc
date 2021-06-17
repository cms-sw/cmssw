//#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "FWCore/Common/interface/OutputProcessBlockHelper.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Common/interface/SubProcessBlockHelper.h"

#include <string>
#include <vector>

TEST_CASE("Test ProcessBlockHelpers", "[ProcessBlockHelpers]") {
  const std::vector<std::string> testNames = {{"HLT"}, {"RECO"}, {"TEST"}};
  const std::vector<std::string> testNames2 = {{"MERGE"}, {"ANA"}, {"HARVEST"}};

  SECTION("OutputProcessBlockHelper") {
    edm::OutputProcessBlockHelper outputProcessBlockHelper;
  }

  SECTION("ProcessBlockHelper") {
    edm::ProcessBlockHelper processBlockHelper;
    processBlockHelper.processesWithProcessBlockProducts() = testNames;
    edm::ProcessBlockHelper const& constProcessBlockHelper = processBlockHelper;
    REQUIRE(processBlockHelper.processesWithProcessBlockProducts() == testNames);
    REQUIRE(constProcessBlockHelper.processesWithProcessBlockProducts() == testNames);

    processBlockHelper.addedProcesses() = testNames2;
    REQUIRE(processBlockHelper.addedProcesses() == testNames2);
    REQUIRE(constProcessBlockHelper.addedProcesses() == testNames2);

    REQUIRE(edm::ProcessBlockHelper::invalidCacheIndex() == 0xffffffff);
    REQUIRE(edm::ProcessBlockHelper::invalidProcessIndex() == 0xffffffff);
  }

  SECTION("SubProcessBlockHelper") { edm::SubProcessBlockHelper subProcessBlockHelper; }
}
