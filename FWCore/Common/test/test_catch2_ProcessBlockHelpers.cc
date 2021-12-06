//#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "FWCore/Common/interface/OutputProcessBlockHelper.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Common/interface/SubProcessBlockHelper.h"

#include <string>
#include <vector>

TEST_CASE("Test ProcessBlockHelpers", "[ProcessBlockHelpers]") {
  const std::string testString("ADD");
  const std::vector<std::string> testNames = {{"HLT"}, {"RECO"}, {"TEST"}};
  const std::vector<std::string> testNames2 = {{"MERGE"}, {"ANA"}, {"HARVEST"}};

  SECTION("OutputProcessBlockHelper") { edm::OutputProcessBlockHelper outputProcessBlockHelper; }

  SECTION("ProcessBlockHelper") {
    edm::ProcessBlockHelper processBlockHelper;
    processBlockHelper.setProcessesWithProcessBlockProducts(testNames);
    REQUIRE(processBlockHelper.processesWithProcessBlockProducts() == testNames);
    processBlockHelper.emplaceBackProcessName(testString);
    std::vector<std::string> testEmplace = testNames;
    testEmplace.emplace_back(testString);
    REQUIRE(processBlockHelper.processesWithProcessBlockProducts() == testEmplace);

    processBlockHelper.setAddedProcesses(testNames2);
    REQUIRE(processBlockHelper.addedProcesses() == testNames2);
    processBlockHelper.emplaceBackAddedProcessName(testString);
    std::vector<std::string> testEmplace2 = testNames2;
    testEmplace2.emplace_back(testString);
    REQUIRE(processBlockHelper.addedProcesses() == testEmplace2);

    REQUIRE(edm::ProcessBlockHelper::invalidCacheIndex() == 0xffffffff);
    REQUIRE(edm::ProcessBlockHelper::invalidProcessIndex() == 0xffffffff);
  }

  SECTION("SubProcessBlockHelper") { edm::SubProcessBlockHelper subProcessBlockHelper; }
}
