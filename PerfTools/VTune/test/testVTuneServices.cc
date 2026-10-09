#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "ittnotify.h"

#include <string>
#include <utility>
#include <vector>

namespace {
  struct IttCalls {
    int pauses = 0;
    int resumes = 0;
    int taskBegins = 0;
    int taskEnds = 0;
    std::string domainName;
    std::string handleName;
    std::vector<std::string> order;
    __itt_domain* beginDomain = nullptr;
    __itt_domain* endDomain = nullptr;
    __itt_string_handle* beginHandle = nullptr;
  } calls;

  __itt_domain domain{1, nullptr, nullptr, 0, nullptr, nullptr};
  __itt_string_handle handle{nullptr, nullptr, 0, nullptr, nullptr};

  void ITTAPI pause() {
    ++calls.pauses;
    calls.order.emplace_back("pause");
  }

  void ITTAPI resume() {
    ++calls.resumes;
    calls.order.emplace_back("resume");
  }

  __itt_domain* ITTAPI createDomain(char const* name) {
    calls.domainName = name;
    calls.order.emplace_back("domain_create");
    return &domain;
  }

  __itt_string_handle* ITTAPI createHandle(char const* name) {
    calls.handleName = name;
    calls.order.emplace_back("string_handle_create");
    return &handle;
  }

  void ITTAPI beginTask(__itt_domain const* taskDomain, __itt_id, __itt_id, __itt_string_handle* taskHandle) {
    ++calls.taskBegins;
    calls.beginDomain = const_cast<__itt_domain*>(taskDomain);
    calls.beginHandle = taskHandle;
    calls.order.emplace_back("task_begin");
  }

  void ITTAPI endTask(__itt_domain const* taskDomain) {
    ++calls.taskEnds;
    calls.endDomain = const_cast<__itt_domain*>(taskDomain);
    calls.order.emplace_back("task_end");
  }

  class IttMock {
  public:
    IttMock()
        : pause_(__itt_pause_ptr),
          resume_(__itt_resume_ptr),
          domainCreate_(__itt_domain_create_ptr),
          handleCreate_(__itt_string_handle_create_ptr),
          taskBegin_(__itt_task_begin_ptr),
          taskEnd_(__itt_task_end_ptr) {
      calls = IttCalls{};
      domain.flags = 1;
      __itt_pause_ptr = pause;
      __itt_resume_ptr = resume;
      __itt_domain_create_ptr = createDomain;
      __itt_string_handle_create_ptr = createHandle;
      __itt_task_begin_ptr = beginTask;
      __itt_task_end_ptr = endTask;
    }

    ~IttMock() {
      __itt_pause_ptr = pause_;
      __itt_resume_ptr = resume_;
      __itt_domain_create_ptr = domainCreate_;
      __itt_string_handle_create_ptr = handleCreate_;
      __itt_task_begin_ptr = taskBegin_;
      __itt_task_end_ptr = taskEnd_;
    }

  private:
    decltype(__itt_pause_ptr) pause_;
    decltype(__itt_resume_ptr) resume_;
    decltype(__itt_domain_create_ptr) domainCreate_;
    decltype(__itt_string_handle_create_ptr) handleCreate_;
    decltype(__itt_task_begin_ptr) taskBegin_;
    decltype(__itt_task_end_ptr) taskEnd_;
  };

  edm::ParameterSet parameters(std::string serviceType, std::vector<std::string> targets) {
    edm::ParameterSet result;
    result.addParameter("@service_type", std::move(serviceType));
    result.addUntrackedParameter("targetModules", std::move(targets));
    return result;
  }

  edm::ServiceToken makeServices(std::vector<edm::ParameterSet> configs, edm::ActivityRegistry& registry) {
    static bool const configured = [] {
      edmplugin::PluginManager::configure(edmplugin::standard::config());
      return true;
    }();
    static_cast<void>(configured);
    auto token = edm::ServiceRegistry::createSet(configs);
    token.copySlotsTo(registry);
    return token;
  }

  edm::ServiceToken makeService(edm::ParameterSet config, edm::ActivityRegistry& registry) {
    return makeServices({std::move(config)}, registry);
  }

  void emitModule(edm::ActivityRegistry& registry, std::string const& moduleName, std::string const& moduleLabel) {
    edm::StreamContext stream{edm::StreamID::invalidStreamID(), nullptr};
    edm::ModuleDescription description{moduleName, moduleLabel};
    edm::ModuleCallingContext context{&description};
    registry.preModuleEventSignal_.emit(stream, context);
    registry.postModuleEventSignal_.emit(stream, context);
  }
}  // namespace

TEST_CASE("VTuneFilterService enables configured modules", "[VTuneFilterService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto service = makeService(parameters("VTuneFilterService", {"selectedLabel", "SelectedType"}), registry);

  SECTION("matches a module label") {
    emitModule(registry, "OtherType", "selectedLabel");
    CHECK(calls.order == std::vector<std::string>{"resume", "pause"});
  }

  SECTION("matches a module type") {
    emitModule(registry, "SelectedType", "otherLabel");
    CHECK(calls.order == std::vector<std::string>{"resume", "pause"});
  }

  SECTION("ignores an unconfigured module") {
    emitModule(registry, "OtherType", "otherLabel");
    CHECK(calls.order.empty());
  }

  SECTION("matches all modules when configured") {
    edm::ActivityRegistry allRegistry;
    auto allService = makeService(parameters("VTuneFilterService", {"all"}), allRegistry);
    emitModule(allRegistry, "OtherType", "otherLabel");
    CHECK(calls.order == std::vector<std::string>{"resume", "pause"});
  }

  SECTION("acts once when label and type both match") {
    emitModule(registry, "SelectedType", "selectedLabel");
    CHECK(calls.resumes == 1);
    CHECK(calls.pauses == 1);
  }
}

TEST_CASE("VTuneFilterService requires targetModules", "[VTuneFilterService]") {
  edm::ParameterSet config;
  config.addParameter("@service_type", std::string{"VTuneFilterService"});
  edm::ActivityRegistry registry;
  REQUIRE_THROWS(makeService(std::move(config), registry));
}

TEST_CASE("VTuneAnnotateService annotates configured modules", "[VTuneAnnotateService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto service = makeService(parameters("VTuneAnnotateService", {"selectedLabel", "SelectedType"}), registry);

  REQUIRE(calls.domainName == "CMSSW.ModuleTracker");

  SECTION("matches a module label") {
    emitModule(registry, "OtherType", "selectedLabel");
    CHECK(calls.handleName == "OtherType/selectedLabel");
    CHECK(calls.beginDomain == &domain);
    CHECK(calls.endDomain == &domain);
    CHECK(calls.beginHandle == &handle);
    CHECK(calls.order == std::vector<std::string>{"domain_create", "string_handle_create", "task_begin", "task_end"});
  }

  SECTION("matches a module type") {
    emitModule(registry, "SelectedType", "otherLabel");
    CHECK(calls.handleName == "SelectedType/otherLabel");
    CHECK(calls.taskBegins == 1);
    CHECK(calls.taskEnds == 1);
  }

  SECTION("ignores an unconfigured module") {
    emitModule(registry, "OtherType", "otherLabel");
    CHECK(calls.order == std::vector<std::string>{"domain_create"});
    CHECK(calls.taskBegins == 0);
    CHECK(calls.taskEnds == 0);
  }

  SECTION("acts once when label and type both match") {
    emitModule(registry, "SelectedType", "selectedLabel");
    CHECK(calls.taskBegins == 1);
    CHECK(calls.taskEnds == 1);
  }
}

TEST_CASE("VTuneAnnotateService requires targetModules", "[VTuneAnnotateService]") {
  IttMock mock;
  edm::ParameterSet config;
  config.addParameter("@service_type", std::string{"VTuneAnnotateService"});
  edm::ActivityRegistry registry;
  REQUIRE_THROWS(makeService(std::move(config), registry));
}

TEST_CASE("VTune services ignore modules with an empty target list", "[VTuneFilterService][VTuneAnnotateService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto services =
      makeServices({parameters("VTuneFilterService", {}), parameters("VTuneAnnotateService", {})}, registry);

  emitModule(registry, "OtherType", "otherLabel");
  CHECK(calls.order == std::vector<std::string>{"domain_create"});
}
