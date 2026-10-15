/*
 *  serviceregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"

#include "catch2/catch_all.hpp"

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <thread>

#include <atomic>
#include <memory>
#include <optional>

namespace {
  struct DummyService {
    int value_;
  };

  struct LifetimeService {
    ~LifetimeService() { destroyed_ = true; }
    static inline bool destroyed_ = false;
  };

  struct UniqueRegistry {
    UniqueRegistry(void* iReg) : otherRegistry_(iReg) {}

    void operator()() { isUnique_ = (otherRegistry_ != &(edm::ServiceRegistry::instance())); }
    void* otherRegistry_;
    static std::atomic<bool> isUnique_;
  };
  std::atomic<bool> UniqueRegistry::isUnique_{false};

  struct PassServices {
    PassServices(edm::ServiceToken iToken, bool& oSucceeded, bool& oCaughtException)
        : token_(iToken), success_(&oSucceeded), caught_(&oCaughtException) {
      *success_ = false;
      *caught_ = false;
    }

    void operator()() {
      try {
        edm::ServiceRegistry::Operate operate(token_);
        edm::Service<testserviceregistry::DummyService> dummy;
        *success_ = dummy->value() == 1;
      } catch (...) {
        *caught_ = true;
      }
    }

    edm::ServiceToken token_;
    bool* success_;
    bool* caught_;
  };
}  // namespace

TEST_CASE("ServiceRegistry", "[ServiceRegistry]") {
  SECTION("loadTest") {
    edm::AssertHandler ah;

    std::vector<edm::ParameterSet> pss;

    edm::ParameterSet ps;
    std::string typeName("DummyService");
    ps.addParameter("@service_type", typeName);
    int value = 2;
    ps.addParameter("value", value);
    pss.push_back(ps);

    edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));

    edm::ServiceRegistry::Operate operate(token);
    edm::Service<testserviceregistry::DummyService> dummy;
    REQUIRE(dummy);
    REQUIRE(dummy.isAvailable());
    REQUIRE(dummy->value() == 2);
  }

  SECTION("externalServiceTest") {
    edm::AssertHandler ah;

    {
      auto dummyPtr = std::make_unique<DummyService>();
      dummyPtr->value_ = 2;
      edm::ServiceToken token(edm::ServiceRegistry::createContaining(std::move(dummyPtr)));

      {
        edm::ServiceRegistry::Operate operate(token);
        edm::Service<DummyService> dummy;
        REQUIRE(dummy);
        REQUIRE(dummy.isAvailable());
        REQUIRE(dummy->value_ == 2);
      }
      {
        auto anotherDummyPtr = std::make_unique<DummyService>();
        anotherDummyPtr->value_ = 4;

        std::vector<edm::ParameterSet> pss;

        edm::ParameterSet ps;
        std::string typeName("DummyService");
        ps.addParameter("@service_type", typeName);
        int value = 3;
        ps.addParameter("value", value);
        pss.push_back(ps);

        edm::ServiceToken tokenFromConfig(edm::ServiceRegistry::createSet(pss));
        edm::ServiceToken anotherToken(edm::ServiceRegistry::createContaining(
            std::move(anotherDummyPtr), tokenFromConfig, edm::serviceregistry::kOverlapIsError));

        edm::ServiceRegistry::Operate operate(anotherToken);
        edm::Service<testserviceregistry::DummyService> dummy;
        REQUIRE(dummy);
        REQUIRE(dummy.isAvailable());
        REQUIRE(dummy->value() == 3);

        edm::Service<DummyService> dummyNotFromConfig;
        REQUIRE(dummyNotFromConfig);
        REQUIRE(dummyNotFromConfig.isAvailable());
        REQUIRE(dummyNotFromConfig->value_ == 4);
      }
    }

    {
      // This is very similar to the above test. Pass in a ServiceWrapper to createContaining instead
      // of a unique_ptr
      auto dummyPtr = std::make_unique<DummyService>();
      auto wrapper = std::make_shared<edm::serviceregistry::ServiceWrapper<DummyService> >(std::move(dummyPtr));

      wrapper->get().value_ = 2;

      {
        edm::ServiceToken token(edm::ServiceRegistry::createContaining(wrapper));
        edm::ServiceRegistry::Operate operate(token);
        edm::Service<DummyService> dummy;
        REQUIRE(dummy);
        REQUIRE(dummy.isAvailable());
        REQUIRE(dummy->value_ == 2);
      }
    }
  }

  SECTION("saveConfigWithExternalTest") {
    //In the HLT the PrescaleService is created once and then used again
    // even if a new EventProcessor is created.  However, the '@save_config' must
    // still be added to the Service's PSet even if that service is not created
    edm::AssertHandler ah;

    std::vector<edm::ParameterSet> pss;

    {
      edm::ParameterSet ps;
      std::string typeName("DummyStoreConfigService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);
    }
    edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));
    REQUIRE(pss[0].exists("@save_config"));

    pss.clear();

    {
      edm::ParameterSet ps;
      std::string typeName("DummyStoreConfigService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);
    }

    //create the services
    edm::ServiceToken token2(edm::ServiceRegistry::createSet(pss, token, edm::serviceregistry::kTokenOverrides));

    REQUIRE(pss[0].exists("@save_config"));
  }

  SECTION("hierarchyTest") {
    edm::AssertHandler ah;

    std::vector<edm::ParameterSet> pss;
    {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);
    }
    edm::ServiceToken token1(edm::ServiceRegistry::createSet(pss));

    pss.clear();
    {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 2;
      ps.addParameter("value", value);
      pss.push_back(ps);
    }
    edm::ServiceToken token2(edm::ServiceRegistry::createSet(pss));

    edm::ServiceRegistry::Operate operate1(token1);
    {
      edm::Service<testserviceregistry::DummyService> dummy;
      REQUIRE(dummy->value() == 1);
    }
    {
      edm::ServiceRegistry::Operate operate2(token2);
      edm::Service<testserviceregistry::DummyService> dummy;
      REQUIRE(dummy->value() == 2);
    }
    {
      edm::Service<testserviceregistry::DummyService> dummy;
      REQUIRE(dummy->value() == 1);
    }
  }

  SECTION("operateNonOwningTest") {
    edm::AssertHandler ah;

    auto makeToken = [](int value) {
      std::vector<edm::ParameterSet> pss;
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      ps.addParameter("value", value);
      pss.push_back(ps);
      return edm::ServiceToken(edm::ServiceRegistry::createSet(pss));
    };
    edm::ServiceToken token1 = makeToken(1);
    edm::ServiceToken token2 = makeToken(2);
    auto value = []() { return edm::Service<testserviceregistry::DummyService>()->value(); };

    // no services on this thread
    REQUIRE_THROWS_AS(edm::Service<testserviceregistry::DummyService>().isAvailable(), cms::Exception);
    {
      edm::ServiceRegistry::OperateNonOwning operate1(token1);
      REQUIRE(value() == 1);
      {
        edm::ServiceRegistry::Operate operate2(token2);
        REQUIRE(value() == 2);
        {
          edm::ServiceRegistry::OperateNonOwning operate1Again(token1);
          REQUIRE(value() == 1);
        }
        REQUIRE(value() == 2);
      }
      REQUIRE(value() == 1);
      {
        edm::ServiceRegistry::OperateNonOwning operate2(token2);
        REQUIRE(value() == 2);
      }
      REQUIRE(value() == 1);
    }
    // no services on this thread
    REQUIRE_THROWS_AS(edm::Service<testserviceregistry::DummyService>().isAvailable(), cms::Exception);
    {
      edm::ServiceRegistry::Operate operate2(token2);
      {
        edm::ServiceRegistry::OperateNonOwning operate1(token1);
        REQUIRE(value() == 1);
      }
      REQUIRE(value() == 2);
    }
    // no services on this thread
    REQUIRE_THROWS_AS(edm::Service<testserviceregistry::DummyService>().isAvailable(), cms::Exception);
  }

  SECTION("operateNonOwningPresentTokenTest") {
    // presentToken() must return an owning token also while the services are made available by OperateNonOwning
    LifetimeService::destroyed_ = false;

    std::optional<edm::ServiceToken> token(edm::ServiceRegistry::createContaining(std::make_unique<LifetimeService>()));
    edm::ServiceToken present;
    {
      edm::ServiceRegistry::OperateNonOwning operate(*token);
      REQUIRE(edm::Service<LifetimeService>().isAvailable());
      present = edm::ServiceRegistry::instance().presentToken();
      {
        // also through an owning Operate nested inside the OperateNonOwning
        edm::ServiceRegistry::Operate operateAgain(*token);
        REQUIRE(edm::Service<LifetimeService>().isAvailable());
      }
      REQUIRE(edm::Service<LifetimeService>().isAvailable());
    }
    // no services on this thread
    REQUIRE_THROWS_AS(edm::Service<LifetimeService>().isAvailable(), cms::Exception);

    token.reset();
    REQUIRE(not LifetimeService::destroyed_);
    {
      edm::ServiceRegistry::Operate operate(present);
      REQUIRE(edm::Service<LifetimeService>().isAvailable());
    }
    present = edm::ServiceToken();
    REQUIRE(LifetimeService::destroyed_);
  }

  SECTION("threadTest") {
    UniqueRegistry::isUnique_ = false;
    void* serviceRegistry = &(edm::ServiceRegistry::instance());
    UniqueRegistry unique(serviceRegistry);
    std::thread testUniqueness(unique);
    testUniqueness.join();
    REQUIRE(UniqueRegistry::isUnique_);

    edm::AssertHandler ah;

    std::vector<edm::ParameterSet> pss;
    {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);
    }
    edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));

    bool succeededToPassServices = false;
    bool exceptionWasThrown = false;

    PassServices passRun(token, succeededToPassServices, exceptionWasThrown);
    std::thread testPassing(passRun);
    testPassing.join();
    REQUIRE(!exceptionWasThrown);
    REQUIRE(succeededToPassServices);
  }
}
