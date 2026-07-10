/*----------------------------------------------------------------------

Test of the EventProcessor class.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"

#include "catch2/catch_all.hpp"

#include "oneapi/tbb/global_control.h"

#include <regex>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

// defined in the other catch2
void doInit();

namespace {
  void work() {
    //std::cout << "work in testeventprocessor" << std::endl;
    std::string configuration(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('p')\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(5))\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.m1 = cms.EDProducer('TestMod',\n"
        "    ivalue = cms.int32(10))\n"
        "process.m2 = cms.EDProducer('TestMod',\n"
        "    ivalue = cms.int32(-3))\n"
        "process.p1 = cms.Path(process.m1*process.m2)\n");
    edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
    proc.beginJob();
    proc.run();
    proc.endJob();
  }
  struct Listener {
    Listener(edm::ActivityRegistry& iAR)
        : postBeginJob_(0),
          postEndJob_(0),
          preEventProcessing_(0),
          postEventProcessing_(0),
          preModule_(0),
          postModule_(0) {
      iAR.watchPostBeginJob(this, &Listener::postBeginJob);
      iAR.watchPostEndJob(this, &Listener::postEndJob);

      iAR.watchPreEvent(this, &Listener::preEventProcessing);
      iAR.watchPostEvent(this, &Listener::postEventProcessing);

      iAR.watchPreModuleEvent(this, &Listener::preModule);
      iAR.watchPostModuleEvent(this, &Listener::postModule);
    }

    void postBeginJob() { ++postBeginJob_; }
    void postEndJob() { ++postEndJob_; }

    void preEventProcessing(edm::StreamContext const&) { ++preEventProcessing_; }
    void postEventProcessing(edm::StreamContext const&) { ++postEventProcessing_; }

    void preModule(edm::StreamContext const&, edm::ModuleCallingContext const&) { ++preModule_; }
    void postModule(edm::StreamContext const&, edm::ModuleCallingContext const&) { ++postModule_; }

    unsigned int postBeginJob_;
    unsigned int postEndJob_;
    unsigned int preEventProcessing_;
    unsigned int postEventProcessing_;
    unsigned int preModule_;
    unsigned int postModule_;
  };
  bool findModuleName(std::string const& iMessage) {
    static std::regex const expr("TestFailuresAnalyzer");
    return regex_search(iMessage, expr);
  }

}  // namespace

TEST_CASE("EventProcessor", "[Framework]") {
  edm::propagate_const<std::unique_ptr<edm::AssertHandler>> m_handler;
  edm::propagate_const<std::unique_ptr<oneapi::tbb::global_control>> m_control;

  if (not m_control) {
    m_control = std::make_unique<oneapi::tbb::global_control>(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  }
  doInit();
  m_handler = std::make_unique<edm::AssertHandler>();

  SECTION("parseTest") {
    try {
      work();
    } catch (cms::Exception& e) {
      std::cerr << "cms exception caught: " << e.explainSelf() << std::endl;
      REQUIRE("Caught cms::Exception " == 0);
    } catch (std::exception& e) {
      std::cerr << "Standard library exception caught: " << e.what() << std::endl;
      REQUIRE("Caught std::exception " == 0);
    } catch (...) {
      REQUIRE("Caught unknown exception " == 0);
    }
  }

  SECTION("beginEndTest") {
    std::string configuration(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('p')\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(10))\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.m1 = cms.EDAnalyzer('TestBeginEndJobAnalyzer')\n"
        "process.p1 = cms.Path(process.m1)\n");
    {
      //std::cout << "beginEndTest 1" << std::endl;
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

      REQUIRE(!TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(0 == proc.totalEvents());

      proc.beginJob();

      //std::cout << "beginEndTest 1 af" << std::endl;

      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(0 == proc.totalEvents());

      proc.endJob();

      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(0 == proc.totalEvents());

      REQUIRE(not edm::pset::Registry::instance()->empty());
    }
    REQUIRE(edm::pset::Registry::instance()->empty());

    {
      //std::cout << "beginEndTest 2" << std::endl;

      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.runToCompletion();

      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());
      REQUIRE(not edm::pset::Registry::instance()->empty());
    }
    REQUIRE(edm::pset::Registry::instance()->empty());
    {
      //std::cout << "beginEndTest 3" << std::endl;

      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.beginJob();
      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);

      // Check that beginJob is not called again
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;

      proc.runToCompletion();

      REQUIRE(!TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());

      proc.endJob();

      REQUIRE(!TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(10 == proc.totalEvents());
    }
    {
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.beginJob();

      // Check that beginJob is not called again
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;

      proc.run();

      REQUIRE(!TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());

      proc.endJob();

      // Check that these are not called again
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;
    }
    REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
    REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);
    {
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.run();

      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());

      // Check that these are not called again
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      proc.endJob();

      REQUIRE(!TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());

      // Check that these are not called again
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;
    }
    REQUIRE(!TestBeginEndJobAnalyzer::control().endRunCalled);
    REQUIRE(!TestBeginEndJobAnalyzer::control().endLumiCalled);

    {
      TestBeginEndJobAnalyzer::control().beginJobCalled = false;
      TestBeginEndJobAnalyzer::control().endJobCalled = false;
      TestBeginEndJobAnalyzer::control().beginRunCalled = false;
      TestBeginEndJobAnalyzer::control().endRunCalled = false;
      TestBeginEndJobAnalyzer::control().beginLumiCalled = false;
      TestBeginEndJobAnalyzer::control().endLumiCalled = false;

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.run();

      REQUIRE(TestBeginEndJobAnalyzer::control().beginJobCalled);
      REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().beginLumiCalled);
      REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
      REQUIRE(10 == proc.totalEvents());
    }
    REQUIRE(!TestBeginEndJobAnalyzer::control().endJobCalled);
    REQUIRE(TestBeginEndJobAnalyzer::control().endRunCalled);
    REQUIRE(TestBeginEndJobAnalyzer::control().endLumiCalled);
  }

  SECTION("cleanupJobTest") {
    //std::cout << "cleanup " << std::endl;
    std::string configuration(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('p')\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(2))\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.m1 = cms.EDAnalyzer('TestBeginEndJobAnalyzer')\n"
        "process.p1 = cms.Path(process.m1)\n");
    {
      //std::cout << "cleanup 1" << std::endl;

      TestBeginEndJobAnalyzer::control().destructorCalled = false;
      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

      REQUIRE(!TestBeginEndJobAnalyzer::control().destructorCalled);
      proc.beginJob();
      REQUIRE(!TestBeginEndJobAnalyzer::control().destructorCalled);
      proc.endJob();
      REQUIRE(!TestBeginEndJobAnalyzer::control().destructorCalled);
    }
    REQUIRE(TestBeginEndJobAnalyzer::control().destructorCalled);
    {
      //std::cout << "cleanup 2" << std::endl;

      TestBeginEndJobAnalyzer::control().destructorCalled = false;
      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

      REQUIRE(!TestBeginEndJobAnalyzer::control().destructorCalled);
      proc.run();
      REQUIRE(2 == proc.totalEvents());
      REQUIRE(!TestBeginEndJobAnalyzer::control().destructorCalled);
    }
    REQUIRE(TestBeginEndJobAnalyzer::control().destructorCalled);
  }

  SECTION("activityRegistryTest") {
    std::string configuration(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('p')\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(5))\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.m1 = cms.EDProducer('TestMod',\n"
        "   ivalue = cms.int32(-3))\n"
        "process.p1 = cms.Path(process.m1)\n");

    std::shared_ptr<edm::ParameterSet> parameterSet = ProcessDescImpl(configuration, false).parameterSet();
    auto processDesc = std::make_shared<edm::ProcessDesc>(parameterSet);

    //We don't want any services, we just want an ActivityRegistry to be created
    // We then use this ActivityRegistry to 'spy on' the signals being produced
    // inside the EventProcessor
    std::vector<edm::ParameterSet> serviceConfigs;
    edm::ServiceToken token = edm::ServiceRegistry::createSet(serviceConfigs);

    edm::ActivityRegistry ar;
    token.connect(ar);
    Listener listener(ar);

    edm::EventProcessor proc(processDesc, token, edm::serviceregistry::kOverlapIsError);

    proc.beginJob();
    proc.run();
    proc.endJob();

    REQUIRE(listener.postBeginJob_ == 1);
    REQUIRE(listener.postEndJob_ == 1);
    REQUIRE(listener.preEventProcessing_ == 5);
    REQUIRE(listener.postEventProcessing_ == 5);
    REQUIRE(listener.preModule_ == 15);
    REQUIRE(listener.postModule_ == 15);

    typedef std::vector<edm::ModuleDescription const*> ModuleDescs;
    ModuleDescs allModules = proc.getAllModuleDescriptions();
    REQUIRE(3 == allModules.size());  // TestMod & TriggerResults
    //std::cout << "\nModuleDescriptions in testeventprocessor::activityRegistryTest()---\n";
    for (ModuleDescs::const_iterator i = allModules.begin(), e = allModules.end(); i != e; ++i) {
      REQUIRE(*i != 0);
      //std::cout << **i << '\n';
    }
    //std::cout << "--- end of ModuleDescriptions\n";

    REQUIRE(5 == proc.totalEvents());
    REQUIRE(5 == proc.totalEventsPassed());
  }

  SECTION("moduleFailureTest") {
    try {
      std::string const preC(
          "import FWCore.ParameterSet.Config as cms\n"
          "process = cms.Process('p')\n"
          "process.maxEvents = cms.untracked.PSet(\n"
          "    input = cms.untracked.int32(2))\n"
          "process.source = cms.Source('EmptySource')\n"
          "process.m1 = cms.EDAnalyzer('TestFailuresAnalyzer',\n"
          "    whichFailure = cms.int32(");

      std::string const postC(
          "))\n"
          "process.p1 = cms.Path(process.m1)\n");

      {
        std::string const configuration = preC + "0" + postC;
        bool threw = true;
        try {
          edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
          threw = false;
        } catch (cms::Exception const& iException) {
          if (!findModuleName(iException.explainSelf())) {
            std::cout << iException.explainSelf() << std::endl;
            REQUIRE(0 == "module name not in exception message");
          }
        }
        REQUIRE(threw);
      }
      {
        std::string const configuration = preC + "1" + postC;
        bool threw = true;
        edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

        try {
          proc.beginJob();
          threw = false;
        } catch (cms::Exception const& iException) {
          if (!findModuleName(iException.explainSelf())) {
            std::cout << iException.explainSelf() << std::endl;
            REQUIRE(0 == "module name not in exception message");
          }
        }
        REQUIRE(threw);
      }

      {
        std::string const configuration = preC + "2" + postC;
        bool threw = true;
        edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

        proc.beginJob();
        try {
          proc.run();
          threw = false;
        } catch (cms::Exception const& iException) {
          if (!findModuleName(iException.explainSelf())) {
            std::cout << iException.explainSelf() << std::endl;
            REQUIRE(0 == "module name not in exception message");
          }
        }
        REQUIRE(threw);
        proc.endJob();
      }
      {
        std::string const configuration = preC + "3" + postC;
        bool threw = true;
        edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

        proc.beginJob();
        try {
          proc.endJob();
          threw = false;
        } catch (cms::Exception const& iException) {
          if (!findModuleName(iException.explainSelf())) {
            std::cout << iException.explainSelf() << std::endl;
            REQUIRE(0 == "module name not in exception message");
          }
        }
        REQUIRE(threw);
      }
      ///
      {
        bool threw = true;
        try {
          std::string configuration(
              "import FWCore.ParameterSet.Config as cms\n"
              "process = cms.Process('p')\n"
              "process.maxEvents = cms.untracked.PSet(\n"
              "    input = cms.untracked.int32(2))\n"
              "process.source = cms.Source('EmptySource')\n"
              "process.p1 = cms.Path(process.m1)\n");
          edm::EventProcessor proc(edm::getPSetFromConfig(configuration));

          threw = false;
        } catch (cms::Exception const& iException) {
          static std::regex const expr("m1");
          if (!regex_search(iException.explainSelf(), expr)) {
            std::cout << iException.explainSelf() << std::endl;
            REQUIRE(0 == "module name not in exception message");
          }
        }
        REQUIRE(threw);
      }
    } catch (cms::Exception const& iException) {
      std::cout << "Unexpected exception " << iException.explainSelf() << std::endl;
      throw;
    }
  }

  SECTION("serviceConfigSaveTest") {
    std::string configuration(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('p')\n"
        "process.add_(cms.Service('DummyStoreConfigService'))\n"
        "process.maxEvents = cms.untracked.PSet(\n"
        "    input = cms.untracked.int32(5))\n"
        "process.source = cms.Source('EmptySource')\n"
        "process.m1 = cms.EDProducer('TestMod',\n"
        "   ivalue = cms.int32(-3))\n"
        "process.p1 = cms.Path(process.m1)\n");

    edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
    edm::ProcessConfiguration const& processConfiguration = proc.processConfiguration();
    edm::ParameterSet const& topPset(edm::getParameterSet(processConfiguration.parameterSetID()));
    REQUIRE(topPset.existsAs<edm::ParameterSet>("DummyStoreConfigService", true));
  }

  SECTION("endpathTest") {}
}
