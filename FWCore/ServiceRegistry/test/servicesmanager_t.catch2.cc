/*
 *  servicesmanager_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/5/05.
 *
 */
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "FWCore/ServiceRegistry/interface/ServicesManager.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "catch2/catch_all.hpp"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include <iostream>

namespace {
  struct DummyService {
    //DummyService(const edm::ParameterSet&,
    //             edm::ActivityRegistry&) {}
  };
}  // namespace

TEST_CASE("ServicesManager", "[ServicesManager]") {
  SECTION("putGetTest") {
    using namespace edm::serviceregistry;

    std::vector<edm::ParameterSet> ps;
    auto sm = ServicesManager::create(ps);

    REQUIRE(!sm->isAvailable<DummyService>());
    bool exceptionThrown = true;
    try {
      sm->get<DummyService>();
      exceptionThrown = false;
    } catch (const edm::Exception&) {
    }
    REQUIRE(exceptionThrown);

    auto ptrWrapper = std::make_shared<ServiceWrapper<DummyService>>(std::make_unique<DummyService>());

    REQUIRE(sm->put(ptrWrapper));

    REQUIRE(sm->isAvailable<DummyService>());

    sm->get<DummyService>();

    REQUIRE(!sm->put(ptrWrapper));
  }

  SECTION("loadTest") {
    typedef testserviceregistry::DummyService TestService;

    using namespace edm::serviceregistry;

    edm::AssertHandler ah;

    {
      std::vector<edm::ParameterSet> pss;

      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);

      auto sm = ServicesManager::create(pss);

      REQUIRE(1 == sm->get<TestService>().value());
    }
    {
      std::vector<edm::ParameterSet> pss;

      edm::ParameterSet ps;
      std::string typeName("DoesntExistService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);

      bool threwConfigurationException = false;
      try {
        auto sm = ServicesManager::create(pss);
      } catch (const cms::Exception&) {
        threwConfigurationException = true;
      }

      REQUIRE(threwConfigurationException);
    }
    {
      std::vector<edm::ParameterSet> pss;

      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);
      pss.push_back(ps);

      bool caughtMultipleServiceError = false;
      try {
        auto sm = ServicesManager::create(pss);
      } catch (const edm::Exception&) {
        caughtMultipleServiceError = true;
      }

      REQUIRE(caughtMultipleServiceError);
    }
    //NEED A TEST FOR SERVICES THAT DEPEND ON OTHER SERVICES
  }

  SECTION("legacyTest") {
    typedef testserviceregistry::DummyService TestService;

    using namespace edm::serviceregistry;

    edm::AssertHandler ah;

    std::string typeName("DummyService");

    std::vector<edm::ParameterSet> pssLegacy;
    {
      edm::ParameterSet ps;
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pssLegacy.push_back(ps);
    }
    auto legacy = ServicesManager::create(pssLegacy);
    REQUIRE(1 == legacy->get<TestService>().value());

    {
      std::vector<edm::ParameterSet> pss;

      edm::ParameterSet ps;
      ps.addParameter("@service_type", typeName);
      int value = 2;
      ps.addParameter("value", value);
      pss.push_back(ps);

      bool threwConfigurationException = false;
      try {
        auto sm = ServicesManager::create(legacy, kOverlapIsError, pss);
      } catch (const edm::Exception&) {
        threwConfigurationException = true;
      }

      REQUIRE(threwConfigurationException);
    }
    {
      std::vector<edm::ParameterSet> pss;

      edm::ParameterSet ps;
      ps.addParameter("@service_type", typeName);
      int value = 2;
      ps.addParameter("value", value);
      pss.push_back(ps);

      auto sm = ServicesManager::create(legacy, kTokenOverrides, pss);

      REQUIRE(1 == sm->get<TestService>().value());
    }
    {
      try {
        std::vector<edm::ParameterSet> pss;

        auto sm = ServicesManager::create(legacy, kOverlapIsError, pss);

        REQUIRE(!sm->get<TestService>().beginJobCalled());
        edm::ActivityRegistry ar;
        sm->connectTo(ar);
        ar.postBeginJobSignal_.emit();

        REQUIRE(sm->get<TestService>().beginJobCalled());
      } catch (const edm::Exception& iException) {
        std::cout << iException.what() << std::endl;
        throw;
      } catch (const std::exception& iException) {
        std::cout << iException.what() << std::endl;
        throw;
      }
    }
  }

  SECTION("dependencyTest") {
    //Try both order of creating services
    typedef testserviceregistry::DummyService TestService;

    using namespace edm::serviceregistry;

    edm::AssertHandler ah;

    {
      std::vector<edm::ParameterSet> pss;
      {
        edm::ParameterSet ps;
        std::string typeName("DummyService");
        ps.addParameter("@service_type", typeName);
        int value = 1;
        ps.addParameter("value", value);
        pss.push_back(ps);
      }
      {
        edm::ParameterSet ps;
        std::string typeName("DependsOnDummyService");
        ps.addParameter("@service_type", typeName);
        pss.push_back(ps);
      }

      auto sm = ServicesManager::create(pss);

      REQUIRE(1 == sm->get<testserviceregistry::DependsOnDummyService>().value());
    }
    {
      std::vector<edm::ParameterSet> pss;
      {
        edm::ParameterSet ps;
        std::string typeName("DependsOnDummyService");
        ps.addParameter("@service_type", typeName);
        pss.push_back(ps);
      }
      {
        edm::ParameterSet ps;
        std::string typeName("DummyService");
        ps.addParameter("@service_type", typeName);
        int value = 1;
        ps.addParameter("value", value);
        pss.push_back(ps);
      }

      auto sm = ServicesManager::create(pss);

      REQUIRE(1 == sm->get<testserviceregistry::DependsOnDummyService>().value());
    }
  }

  SECTION("saveConfigTest") {
    typedef testserviceregistry::DummyService TestService;

    using namespace edm::serviceregistry;

    edm::AssertHandler ah;

    {
      std::vector<edm::ParameterSet> pss;
      {
        edm::ParameterSet ps;
        std::string typeName("DummyService");
        ps.addParameter("@service_type", typeName);
        int value = 1;
        ps.addParameter("value", value);
        pss.push_back(ps);
      }

      {
        edm::ParameterSet ps;
        std::string typeName("DummyStoreConfigService");
        ps.addParameter("@service_type", typeName);
        pss.push_back(ps);
      }

      auto sm = ServicesManager::create(pss);

      REQUIRE(!pss[0].exists("@save_config"));
      REQUIRE(pss[1].exists("@save_config"));
    }
  }
}
