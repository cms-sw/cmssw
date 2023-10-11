#include "catch.hpp"

#include "FWCore/Framework/interface/CmsRunParser.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <string>

TEST_CASE("test cmsRun command line parsing", "[commandline]") {
  edm::CmsRunParser parser("cmsRun");

  SECTION("No arguments") {
    constexpr int kSize = 1;
    const char* args[kSize] = {"cmsRun"};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(edm::CmsRunParser::hasVM(output));
    auto vm = edm::CmsRunParser::getVM(output);

    REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kParameterSetOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kPythonOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
  }

  SECTION("help") {
    constexpr int kSize = 2;
    const char* args[kSize] = {"cmsRun", "-h"};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(not edm::CmsRunParser::hasVM(output));
    REQUIRE(edm::CmsRunParser::getExit(output) == 0);
  }

  SECTION("wrong") {
    constexpr int kSize = 2;
    const char* args[kSize] = {"cmsRun", "--wrong"};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(not edm::CmsRunParser::hasVM(output));
    REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
  }

  SECTION("Config file only") {
    constexpr int kSize = 2;
    const std::string arg("config.py");
    const char* args[kSize] = {"cmsRun", arg.c_str()};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(edm::CmsRunParser::hasVM(output));
    auto vm = edm::CmsRunParser::getVM(output);

    REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
    REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
    REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kPythonOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
  }

  SECTION("Command line input only") {
    constexpr int kSize = 3;
    const std::string arg(
        "import FWCore.ParameterSet.Config as cms; process = cms.Process('Test'); "
        "process.source=cms.Source('EmptySource'); process.maxEvents.input=10; print('Test3')");
    const char* args[kSize] = {"cmsRun", "-c", arg.c_str()};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(edm::CmsRunParser::hasVM(output));
    auto vm = edm::CmsRunParser::getVM(output);

    REQUIRE(not vm.count(edm::CmsRunParser::kParameterSetOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
    REQUIRE(not vm.count(edm::CmsRunParser::kPythonOpt));
    REQUIRE(vm.count(edm::CmsRunParser::kCmdOpt));
    REQUIRE(vm[edm::CmsRunParser::kCmdOpt].as<std::string>() == arg);
  }
}
