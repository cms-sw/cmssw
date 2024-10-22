#include "catch.hpp"

#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "TestTypeResolvers.h"

#include <iostream>

namespace edm::test {
  class FactoryTestAProd : public edm::global::EDProducer<> {
  public:
    explicit FactoryTestAProd(edm::ParameterSet const&) { ++count_; }
    void produce(StreamID, edm::Event&, edm::EventSetup const&) const final {}
    static int count_;
  };
  int FactoryTestAProd::count_ = 0;

  namespace cpu {
    class FactoryTestAProd : public edm::global::EDProducer<> {
    public:
      explicit FactoryTestAProd(edm::ParameterSet const&) { ++count_; }
      void produce(StreamID, edm::Event&, edm::EventSetup const&) const final {}
      static int count_;
    };
    int FactoryTestAProd::count_ = 0;
  }  // namespace cpu
  namespace other {
    class FactoryTestAProd : public edm::global::EDProducer<> {
    public:
      explicit FactoryTestAProd(edm::ParameterSet const&) { ++count_; }
      void produce(StreamID, edm::Event&, edm::EventSetup const&) const final {}
      static int count_;
    };
    int FactoryTestAProd::count_ = 0;
  }  // namespace other
}  // namespace edm::test

DEFINE_FWK_MODULE(edm::test::FactoryTestAProd);
namespace edm::test {
  using FactoryTestBProd = FactoryTestAProd;
}
DEFINE_FWK_MODULE(edm::test::FactoryTestBProd);
DEFINE_FWK_MODULE(edm::test::cpu::FactoryTestAProd);
DEFINE_FWK_MODULE(edm::test::other::FactoryTestAProd);
namespace edm::test::cpu {
  using FactoryTestCProd = FactoryTestAProd;
}
DEFINE_FWK_MODULE(edm::test::cpu::FactoryTestCProd);

static bool called = false;
using namespace edm;
TEST_CASE("test edm::Factory", "[Factory]") {
  signalslot::Signal<void(const ModuleDescription&)> pre;
  signalslot::Signal<void(const ModuleDescription&)> post;
  ProductRegistry prodReg;
  PreallocationConfiguration preallocConfig;
  std::shared_ptr<ProcessConfiguration const> procConfig = std::make_shared<ProcessConfiguration>();
  if (not called) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    called = true;
  }

  SECTION("test missing plugin") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "DoesNotExistModule");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");

    CHECK_THROWS(
        factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), nullptr, pre, post));
    try {
      factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), nullptr, pre, post);
    } catch (cms::Exception const& iE) {
      REQUIRE(std::string(iE.what()).find("DoesNotExistModule") != std::string::npos);
    }
  }
  SECTION("test missing plugin with simple resolver") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "DoesNotExistModule");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    edm::test::SimpleTestTypeResolverMaker resolver;
    using Catch::Matchers::Contains;
    CHECK_THROWS_WITH(
        factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post),
        Contains("DoesNotExistModule"));
  }
  SECTION("test missing plugin with complex resolver") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "generic::DoesNotExistModule");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    edm::test::ComplexTestTypeResolverMaker resolver;
    using Catch::Matchers::Contains;
    CHECK_THROWS_WITH(
        factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post),
        Contains("generic::DoesNotExistModule") && Contains("edm::test::other::DoesNotExistModule") &&
            Contains("edm::test::cpu::DoesNotExistModule"));
  }
  SECTION("test missing plugin with configurable resolver") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "generic::DoesNotExistModule");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    SECTION("default behavior") {
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      using Catch::Matchers::Contains;
      CHECK_THROWS_WITH(
          factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post),
          Contains("generic::DoesNotExistModule") && Contains("edm::test::other::DoesNotExistModule") &&
              Contains("edm::test::cpu::DoesNotExistModule"));
    }
    SECTION("set variant to other") {
      pset.addUntrackedParameter<std::string>("variant", "other");
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      using Catch::Matchers::Contains;
      CHECK_THROWS_WITH(
          factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post),
          Contains("generic::DoesNotExistModule") && Contains("edm::test::other::DoesNotExistModule") &&
              not Contains("edm::test::cpu::DoesNotExistModule"));
    }
    SECTION("set variant to cpu") {
      pset.addUntrackedParameter<std::string>("variant", "cpu");
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      using Catch::Matchers::Contains;
      CHECK_THROWS_WITH(
          factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post),
          Contains("generic::DoesNotExistModule") && not Contains("edm::test::other::DoesNotExistModule") &&
              Contains("edm::test::cpu::DoesNotExistModule"));
    }
  }

  SECTION("test found plugin") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "edm::test::FactoryTestAProd");
    pset.addParameter<std::string>("@module_label", "a");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");

    REQUIRE(edm::test::FactoryTestAProd::count_ == 0);
    REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), nullptr, pre, post));
    REQUIRE(edm::test::FactoryTestAProd::count_ == 1);
    edm::test::FactoryTestAProd::count_ = 0;
  }
  SECTION("test found plugin with simple resolver") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "edm::test::FactoryTestBProd");
    pset.addParameter<std::string>("@module_label", "b");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    edm::test::SimpleTestTypeResolverMaker resolver;
    REQUIRE(edm::test::FactoryTestBProd::count_ == 0);
    REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
    REQUIRE(edm::test::FactoryTestBProd::count_ == 1);
    edm::test::FactoryTestBProd::count_ = 0;
  }
  SECTION("test found plugin with complex resolver") {
    SECTION("find other") {
      auto factory = Factory::get();
      ParameterSet pset;
      pset.addParameter<std::string>("@module_type", "generic::FactoryTestAProd");
      pset.addParameter<std::string>("@module_label", "gen");
      pset.addParameter<std::string>("@module_edm_type", "EDProducer");
      edm::test::ComplexTestTypeResolverMaker resolver;
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 0);
      REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 1);
      edm::test::other::FactoryTestAProd::count_ = 0;
    }
    SECTION("find cpu") {
      auto factory = Factory::get();
      ParameterSet pset;
      pset.addParameter<std::string>("@module_type", "generic::FactoryTestCProd");
      pset.addParameter<std::string>("@module_label", "cgen");
      pset.addParameter<std::string>("@module_edm_type", "EDProducer");
      edm::test::ComplexTestTypeResolverMaker resolver;
      REQUIRE(edm::test::cpu::FactoryTestCProd::count_ == 0);
      REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
      REQUIRE(edm::test::cpu::FactoryTestCProd::count_ == 1);
      edm::test::cpu::FactoryTestCProd::count_ = 0;
    }
  }
  SECTION("test found plugin with configurable resolver") {
    auto factory = Factory::get();
    ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "generic::FactoryTestAProd");
    pset.addParameter<std::string>("@module_label", "gen");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    SECTION("default behavior") {
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 0);
      REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 1);
      edm::test::other::FactoryTestAProd::count_ = 0;
    }
    SECTION("set variant to cpu") {
      pset.addUntrackedParameter<std::string>("variant", "cpu");
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 0);
      REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 1);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 0);
      edm::test::cpu::FactoryTestAProd::count_ = 0;
    }
    SECTION("set variant to other") {
      pset.addUntrackedParameter<std::string>("variant", "other");
      edm::test::ConfigurableTestTypeResolverMaker resolver;
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 0);
      REQUIRE(factory->makeModule(MakeModuleParams(&pset, prodReg, &preallocConfig, procConfig), &resolver, pre, post));
      REQUIRE(edm::test::cpu::FactoryTestAProd::count_ == 0);
      REQUIRE(edm::test::other::FactoryTestAProd::count_ == 1);
      edm::test::other::FactoryTestAProd::count_ = 0;
    }
  }
}
