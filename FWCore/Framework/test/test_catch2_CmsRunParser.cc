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

  SECTION("wrong (short)") {
    constexpr int kSize = 2;
    const char* args[kSize] = {"cmsRun", "-w"};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(not edm::CmsRunParser::hasVM(output));
    REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
  }

  SECTION("wrong (long)") {
    constexpr int kSize = 2;
    const char* args[kSize] = {"cmsRun", "--wrong"};
    const auto& output = parser.parse(kSize, args);

    REQUIRE(not edm::CmsRunParser::hasVM(output));
    REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
  }

  SECTION("Config file") {
    const std::string arg = "config.py";

    SECTION("Config file only") {
      constexpr int kSize = 2;
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

    SECTION("With --") {
      constexpr int kSize = 3;
      const char* args[kSize] = {"cmsRun", "--", arg.c_str()};
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

    SECTION("cmsRun argument") {
      constexpr int kSize = 4;
      const std::string jobreport = "report.xml";
      const char* args[kSize] = {"cmsRun", "-j", jobreport.c_str(), arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(vm[edm::CmsRunParser::kJobreportOpt].as<std::string>() == jobreport);
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kPythonOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("Wrong cmsRun argument (short)") {
      constexpr int kSize = 3;
      const char* args[kSize] = {"cmsRun", "-w", arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(not edm::CmsRunParser::hasVM(output));
      REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
    }

    SECTION("Wrong cmsRun argument (long)") {
      constexpr int kSize = 3;
      const char* args[kSize] = {"cmsRun", "--wrong", arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(not edm::CmsRunParser::hasVM(output));
      REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
    }

    SECTION("python argument") {
      constexpr int kSize = 3;
      const std::string parg = "--test";
      const char* args[kSize] = {"cmsRun", arg.c_str(), parg.c_str()};
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
      REQUIRE(vm.count(edm::CmsRunParser::kPythonOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("cmsRun and python argument") {
      constexpr int kSize = 5;
      const std::string jobreport = "report.xml";
      const std::string parg = "--test";
      const char* args[kSize] = {"cmsRun", "-j", jobreport.c_str(), arg.c_str(), parg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(vm[edm::CmsRunParser::kJobreportOpt].as<std::string>() == jobreport);
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("Same cmsRun and python argument") {
      constexpr int kSize = 4;
      const std::string parg = "--strict";
      const char* args[kSize] = {"cmsRun", parg.c_str(), arg.c_str(), parg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }
  }

  SECTION("Config file with -") {
    const std::string arg = "-config.py";

    SECTION("Config file only") {
      constexpr int kSize = 3;
      const char* args[kSize] = {"cmsRun", "--", arg.c_str()};
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

    SECTION("cmsRun argument") {
      constexpr int kSize = 5;
      const std::string jobreport = "report.xml";
      const char* args[kSize] = {"cmsRun", "-j", jobreport.c_str(), "--", arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(vm[edm::CmsRunParser::kJobreportOpt].as<std::string>() == jobreport);
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kPythonOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("Wrong cmsRun argument (short)") {
      constexpr int kSize = 4;
      const char* args[kSize] = {"cmsRun", "-w", "--", arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(not edm::CmsRunParser::hasVM(output));
      REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
    }

    SECTION("Wrong cmsRun argument (long)") {
      constexpr int kSize = 4;
      const char* args[kSize] = {"cmsRun", "--wrong", "--", arg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(not edm::CmsRunParser::hasVM(output));
      REQUIRE(edm::CmsRunParser::getExit(output) == edm::errors::CommandLineProcessing);
    }

    SECTION("python argument") {
      constexpr int kSize = 4;
      const std::string parg = "--test";
      const char* args[kSize] = {"cmsRun", "--", arg.c_str(), parg.c_str()};
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
      REQUIRE(vm.count(edm::CmsRunParser::kPythonOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("cmsRun and python arguments") {
      constexpr int kSize = 6;
      const std::string jobreport = "report.xml";
      const std::string parg = "--test";
      const char* args[kSize] = {"cmsRun", "-j", jobreport.c_str(), "--", arg.c_str(), parg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(vm[edm::CmsRunParser::kJobreportOpt].as<std::string>() == jobreport);
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }

    SECTION("same cmsRun and python arguments") {
      constexpr int kSize = 5;
      const std::string parg = "--strict";
      const char* args[kSize] = {"cmsRun", parg.c_str(), "--", arg.c_str(), parg.c_str()};
      const auto& output = parser.parse(kSize, args);

      REQUIRE(edm::CmsRunParser::hasVM(output));
      auto vm = edm::CmsRunParser::getVM(output);

      REQUIRE(vm.count(edm::CmsRunParser::kParameterSetOpt));
      REQUIRE(vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>() == arg);
      REQUIRE(not vm.count(edm::CmsRunParser::kHelpOpt));
      REQUIRE(vm.count(edm::CmsRunParser::kStrictOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kEnableJobreportOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kNumberOfThreadsOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt));
      REQUIRE(not vm.count(edm::CmsRunParser::kJobModeOpt));
      auto pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      REQUIRE(pythonOptValues.size() == 1);
      REQUIRE(pythonOptValues[0] == parg);
      REQUIRE(not vm.count(edm::CmsRunParser::kCmdOpt));
    }
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
