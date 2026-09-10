#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "PerfTools/VtuneFilter/interface/VtuneAnnotateService.h"
#include "PerfTools/VtuneFilter/interface/VtuneFilterService.h"

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

  edm::ParameterSet parameters(std::vector<std::string> targets) {
    edm::ParameterSet result;
    result.addUntrackedParameter("targetModules", std::move(targets));
    return result;
  }

  template <typename Service>
  void emitModule(edm::ActivityRegistry& registry,
                  Service&,
                  std::string const& moduleName,
                  std::string const& moduleLabel) {
    edm::StreamContext stream{edm::StreamID::invalidStreamID(), nullptr};
    edm::ModuleDescription description{moduleName, moduleLabel};
    edm::ModuleCallingContext context{&description};
    registry.preModuleEventSignal_.emit(stream, context);
    registry.postModuleEventSignal_.emit(stream, context);
  }
}  // namespace

TEST_CASE("VtuneFilterService filters configured modules", "[VtuneFilterService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto config = parameters({"selectedLabel", "SelectedType"});
  edm::VtuneFilterService service{config, registry};

  SECTION("matches a module label") {
    emitModule(registry, service, "OtherType", "selectedLabel");
    CHECK(calls.order == std::vector<std::string>{"pause", "resume"});
  }

  SECTION("matches a module type") {
    emitModule(registry, service, "SelectedType", "otherLabel");
    CHECK(calls.order == std::vector<std::string>{"pause", "resume"});
  }

  SECTION("ignores an unconfigured module") {
    emitModule(registry, service, "OtherType", "otherLabel");
    CHECK(calls.order.empty());
  }

  SECTION("acts once when label and type both match") {
    emitModule(registry, service, "SelectedType", "selectedLabel");
    CHECK(calls.pauses == 1);
    CHECK(calls.resumes == 1);
  }
}

TEST_CASE("VtuneFilterService requires targetModules", "[VtuneFilterService]") {
  edm::ParameterSet config;
  edm::ActivityRegistry registry;
  REQUIRE_THROWS(edm::VtuneFilterService(config, registry));
}

TEST_CASE("VtuneAnnotateService annotates configured modules", "[VtuneAnnotateService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto config = parameters({"selectedLabel", "SelectedType"});
  edm::VtuneAnnotateService service{config, registry};

  REQUIRE(calls.domainName == "CMSSW.ModuleTracker");

  SECTION("matches a module label") {
    emitModule(registry, service, "OtherType", "selectedLabel");
    CHECK(calls.handleName == "OtherType/selectedLabel");
    CHECK(calls.beginDomain == &domain);
    CHECK(calls.endDomain == &domain);
    CHECK(calls.beginHandle == &handle);
    CHECK(calls.order == std::vector<std::string>{"domain_create", "string_handle_create", "task_begin", "task_end"});
  }

  SECTION("matches a module type") {
    emitModule(registry, service, "SelectedType", "otherLabel");
    CHECK(calls.handleName == "SelectedType/otherLabel");
    CHECK(calls.taskBegins == 1);
    CHECK(calls.taskEnds == 1);
  }

  SECTION("ignores an unconfigured module") {
    emitModule(registry, service, "OtherType", "otherLabel");
    CHECK(calls.order == std::vector<std::string>{"domain_create"});
    CHECK(calls.taskBegins == 0);
    CHECK(calls.taskEnds == 0);
  }

  SECTION("acts once when label and type both match") {
    emitModule(registry, service, "SelectedType", "selectedLabel");
    CHECK(calls.taskBegins == 1);
    CHECK(calls.taskEnds == 1);
  }
}

TEST_CASE("VtuneAnnotateService requires targetModules", "[VtuneAnnotateService]") {
  IttMock mock;
  edm::ParameterSet config;
  edm::ActivityRegistry registry;
  REQUIRE_THROWS(edm::VtuneAnnotateService(config, registry));
}

TEST_CASE("Vtune services ignore modules with an empty target list", "[VtuneFilterService][VtuneAnnotateService]") {
  IttMock mock;
  edm::ActivityRegistry registry;
  auto config = parameters({});
  edm::VtuneFilterService filter{config, registry};
  edm::VtuneAnnotateService annotate{config, registry};

  emitModule(registry, filter, "OtherType", "otherLabel");
  CHECK(calls.order == std::vector<std::string>{"domain_create"});
}
