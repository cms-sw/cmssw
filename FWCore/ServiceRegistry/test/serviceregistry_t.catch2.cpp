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
#include "FWCore/ServiceRegistry/test/stubs/TokenHolderService.h"

#include "catch2/catch_all.hpp"

#include "FWCore/PluginManager/interface/ProblemTracker.h"

#include <thread>

#include <atomic>
#include <type_traits>

namespace {
  struct DummyService {
    int value_;
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

// a ServiceToken is a non-owning handle, as cheap to copy as a pointer
static_assert(std::is_trivially_copyable_v<edm::ServiceToken>);
static_assert(sizeof(edm::ServiceToken) == sizeof(void*));
// it can be obtained from an OwningServiceToken, but not from a temporary one, as it would dangle
static_assert(std::is_convertible_v<edm::OwningServiceToken const&, edm::ServiceToken>);
static_assert(not std::is_convertible_v<edm::OwningServiceToken&&, edm::ServiceToken>);
// and it cannot be turned back into an owning token implicitly
static_assert(not std::is_convertible_v<edm::ServiceToken, edm::OwningServiceToken>);

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

    edm::OwningServiceToken token(edm::ServiceRegistry::createSet(pss));

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
      edm::OwningServiceToken token(edm::ServiceRegistry::createContaining(std::move(dummyPtr)));

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

        edm::OwningServiceToken tokenFromConfig(edm::ServiceRegistry::createSet(pss));
        edm::OwningServiceToken anotherToken(edm::ServiceRegistry::createContaining(
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
        edm::OwningServiceToken token(edm::ServiceRegistry::createContaining(wrapper));
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
    edm::OwningServiceToken token(edm::ServiceRegistry::createSet(pss));
    REQUIRE(pss[0].exists("@save_config"));

    pss.clear();

    {
      edm::ParameterSet ps;
      std::string typeName("DummyStoreConfigService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);
    }

    //create the services
    edm::OwningServiceToken token2(edm::ServiceRegistry::createSet(pss, token, edm::serviceregistry::kTokenOverrides));

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
    edm::OwningServiceToken token1(edm::ServiceRegistry::createSet(pss));

    pss.clear();
    {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 2;
      ps.addParameter("value", value);
      pss.push_back(ps);
    }
    edm::OwningServiceToken token2(edm::ServiceRegistry::createSet(pss));

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
    edm::OwningServiceToken token(edm::ServiceRegistry::createSet(pss));

    bool succeededToPassServices = false;
    bool exceptionWasThrown = false;

    PassServices passRun(token, succeededToPassServices, exceptionWasThrown);
    std::thread testPassing(passRun);
    testPassing.join();
    REQUIRE(!exceptionWasThrown);
    REQUIRE(succeededToPassServices);
  }

  SECTION("tokenTest") {
    edm::AssertHandler ah;

    REQUIRE(not edm::ServiceToken());
    REQUIRE(not edm::OwningServiceToken());
    REQUIRE(not edm::ServiceRegistry::instance().presentToken());

    auto dummyPtr = std::make_unique<DummyService>();
    dummyPtr->value_ = 3;
    auto owner = std::make_unique<edm::OwningServiceToken>(edm::ServiceRegistry::createContaining(std::move(dummyPtr)));
    edm::ServiceToken token = *owner;
    REQUIRE(token);
    REQUIRE(token == owner->token());

    edm::WeakServiceToken weak(token);
    edm::OwningServiceToken shared;
    {
      edm::ServiceRegistry::Operate operate(token);
      REQUIRE(edm::ServiceRegistry::instance().presentToken() == token);
      // share the ownership of the services in use
      shared = edm::OwningServiceToken(edm::ServiceRegistry::instance().presentToken());
    }
    REQUIRE(not edm::ServiceRegistry::instance().presentToken());

    // the services are kept alive by the other owner
    owner.reset();
    {
      auto locked = weak.lock();
      REQUIRE(locked);
      edm::ServiceRegistry::Operate operate(locked);
      edm::Service<DummyService> dummy;
      REQUIRE(dummy->value_ == 3);
    }

    // until the last owner is gone
    shared = edm::OwningServiceToken();
    REQUIRE(not weak.lock());
  }

  SECTION("tokenInServiceConstructorTest") {
    edm::AssertHandler ah;

    std::vector<edm::ParameterSet> pss;
    edm::ParameterSet ps;
    ps.addParameter("@service_type", std::string("TokenHolderService"));
    pss.push_back(ps);

    // the services are created once the ServicesManager is owned, so a service can take an owning or a weak
    // token to its own service set in its constructor
    edm::OwningServiceToken owner(edm::ServiceRegistry::createSet(pss));
    edm::WeakServiceToken weak;
    {
      edm::ServiceRegistry::Operate operate(owner);
      edm::Service<testserviceregistry::TokenHolderService> holder;
      REQUIRE(holder.isAvailable());
      REQUIRE(holder->ownedAtConstruction());
      weak = holder->weakToken();
      auto locked = weak.lock();
      REQUIRE(locked.token() == owner.token());
    }

    // the weak token expires with the owner
    owner = edm::OwningServiceToken();
    REQUIRE(not weak.lock());
  }
}
