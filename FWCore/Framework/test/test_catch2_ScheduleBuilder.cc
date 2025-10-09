#include "catch2/catch_all.hpp"

#include "FWCore/Framework/src/ScheduleBuilder.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"

#include <vector>
#include <string_view>
namespace edm::test::sb {
  class SBIntProducer : public edm::global::EDProducer<> {
  public:
    explicit SBIntProducer(std::vector<edm::InputTag> const& iConsumes) {
      for (auto const& tag : iConsumes) {
        consumes<int>(tag);
      }
      produces<int>();
    }
    void produce(StreamID, edm::Event&, edm::EventSetup const&) const final {}
  };
}  // namespace edm::test::sb

namespace {
  struct ModuleInfo {
    std::string label_;
    std::string type_;
    std::string edmType_;
  };
  enum class EdmType { kProducer, kFilter, kAnalyzer, kOutputModule };

  struct ConfigBuilder {
    void addModule(std::string const& iLabel, std::string const& iType, EdmType iEdmType) {
      ModuleInfo info = {.label_ = iLabel, .type_ = iType, .edmType_ = enumToName(iEdmType)};
      modules_.emplace_back(std::move(info));
    }
    std::string enumToName(EdmType iType) {
      switch (iType) {
        case EdmType::kProducer:
          return "EDProducer";
        case EdmType::kFilter:
          return "EDFilter";
        case EdmType::kAnalyzer:
          return "EDAnalyzer";
        case EdmType::kOutputModule:
          return "OutputModule";
      }
      return "";
    }
    void addPath(std::string iName, std::vector<std::string> iModuleLabels) {
      paths_.emplace_back(std::move(iName), std::move(iModuleLabels));
    }
    void addEndPath(std::string iName, std::vector<std::string> iModuleLabels) {
      endpaths_.emplace_back(std::move(iName), std::move(iModuleLabels));
    }
    void addAllowedFilterOnEndPath(std::string iLabel) { allowedFiltersOnEndPaths_.emplace_back(std::move(iLabel)); }

    edm::ParameterSet buildConfig() {
      edm::ParameterSet config;
      config.addParameter<std::vector<std::string>>("@all_aliases", std::vector<std::string>());
      std::vector<std::string> pathNames;
      for (auto const& pathInfo : paths_) {
        pathNames.emplace_back(pathInfo.first);
        config.addParameter<std::vector<std::string>>(pathInfo.first, pathInfo.second);
      }
      for (auto const& pathInfo : endpaths_) {
        config.addParameter<std::vector<std::string>>(pathInfo.first, pathInfo.second);
      }
      edm::ParameterSet trigPaths;
      trigPaths.addParameter<std::vector<std::string>>("@trigger_paths", pathNames);
      config.addParameter<edm::ParameterSet>("@trigger_paths", trigPaths);
      std::vector<std::string> moduleLabels;
      moduleLabels.reserve(modules_.size());
      for (auto const& moduleInfo : modules_) {
        moduleLabels.emplace_back(moduleInfo.label_);
        edm::ParameterSet moduleConfig;
        moduleConfig.addParameter<std::string>("@module_type", moduleInfo.type_);
        moduleConfig.addParameter<std::string>("@module_edm_type", moduleInfo.edmType_);
        config.addParameter<edm::ParameterSet>(moduleInfo.label_, moduleConfig);
      }
      config.addParameter<std::vector<std::string>>("@all_modules", moduleLabels);
      config.addUntrackedParameter<std::vector<std::string>>("@filters_on_endpaths", allowedFiltersOnEndPaths_);

      return config;
    }

    std::vector<std::string> paths() {
      std::vector<std::string> ret;
      for (auto const& p : paths_) {
        ret.push_back(p.first);
      }
      return ret;
    }

    std::vector<std::string> endpaths() {
      std::vector<std::string> ret;
      for (auto const& p : endpaths_) {
        ret.push_back(p.first);
      }
      return ret;
    }

  private:
    std::vector<ModuleInfo> modules_;
    std::vector<std::pair<std::string, std::vector<std::string>>> paths_;
    std::vector<std::pair<std::string, std::vector<std::string>>> endpaths_;
    std::vector<std::string> allowedFiltersOnEndPaths_;
  };
}  // namespace

TEST_CASE("test edm::ScheduleBuilder", "[ScheduleBuilder]") {
  edm::HardwareResourcesDescription hardware;
  auto const processConfig = std::make_shared<edm::ProcessConfiguration const>("TEST", "CMSSW_X_Y_Z", hardware);
  edm::PreallocationConfiguration prealloc;
  SECTION("empty schedule") {
    edm::ModuleRegistry moduleRegistry;
    edm::SignallingProductRegistryFiller productRegistry;
    edm::ActivityRegistry activityRegistry;

    ConfigBuilder confBuilder;
    auto config = confBuilder.buildConfig();
    edm::ScheduleBuilder builder(moduleRegistry,
                                 config,
                                 confBuilder.paths(),
                                 confBuilder.endpaths(),
                                 prealloc,
                                 productRegistry,
                                 activityRegistry,
                                 processConfig);

    REQUIRE(builder.endpathNameAndModules_.empty());
    REQUIRE(builder.pathNameAndModules_.empty());
    REQUIRE(builder.allNeededModules_.empty());
    REQUIRE(builder.unscheduledModules_.empty());
    REQUIRE(builder.pathStatusInserters_.empty());
    REQUIRE(not builder.resultsInserter_);
  }
  SECTION("one module on path") {
    edm::SignallingProductRegistryFiller productRegistry;
    edm::ActivityRegistry activityRegistry;

    edm::ModuleRegistry moduleRegistry;
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "edm::test::sb::SBIntProducer");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    pset.registerIt();

    edm::ModuleDescription desc(
        pset.id(), "edm::test::sb::SBIntProducer", "prod1", processConfig.get(), edm::ModuleDescription::getUniqueID());
    moduleRegistry.makeExplicitModule<edm::test::sb::SBIntProducer>(desc,
                                                                    prealloc,
                                                                    &productRegistry,
                                                                    activityRegistry.preModuleConstructionSignal_,
                                                                    activityRegistry.postModuleConstructionSignal_,
                                                                    std::vector<edm::InputTag>());

    ConfigBuilder confBuilder;
    confBuilder.addModule("prod1", "edm::test::sb::SBIntProducer", EdmType::kProducer);
    confBuilder.addPath("p1", {{"prod1"}});
    auto config = confBuilder.buildConfig();
    edm::ScheduleBuilder builder(moduleRegistry,
                                 config,
                                 confBuilder.paths(),
                                 confBuilder.endpaths(),
                                 prealloc,
                                 productRegistry,
                                 activityRegistry,
                                 processConfig);

    REQUIRE(builder.endpathNameAndModules_.empty());
    REQUIRE(builder.pathNameAndModules_.size() == 1);
    REQUIRE(builder.pathNameAndModules_.front().first == "p1");
    REQUIRE(builder.pathNameAndModules_.front().second.size() == 1);
    auto moduleInfo = builder.pathNameAndModules_.front().second.front();
    REQUIRE(moduleInfo.description_->moduleLabel() == "prod1");
    REQUIRE(moduleInfo.description_->moduleName() == "edm::test::sb::SBIntProducer");
    REQUIRE(moduleInfo.action_ == edm::WorkerInPath::Normal);
    REQUIRE(moduleInfo.placeInPath_ == 0);
    REQUIRE(moduleInfo.runConcurrently_ == true);
    REQUIRE(builder.allNeededModules_.size() == 3);
    REQUIRE(std::find_if(builder.allNeededModules_.begin(), builder.allNeededModules_.end(), [&desc](auto const* el) {
              return *el == desc;
            }) != builder.allNeededModules_.end());
    REQUIRE(builder.unscheduledModules_.empty());
    REQUIRE(builder.pathStatusInserters_.size() == 1);
    REQUIRE(builder.endPathStatusInserters_.empty());
    REQUIRE(builder.resultsInserter_);
  }

  SECTION("one module on endpath") {
    //If there is only one endpath no end path status inserter is added to the system
    edm::SignallingProductRegistryFiller productRegistry;
    edm::ActivityRegistry activityRegistry;

    edm::ModuleRegistry moduleRegistry;
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@module_type", "edm::test::sb::SBIntProducer");
    pset.addParameter<std::string>("@module_edm_type", "EDProducer");
    pset.registerIt();

    edm::ModuleDescription desc(
        pset.id(), "edm::test::sb::SBIntProducer", "prod1", processConfig.get(), edm::ModuleDescription::getUniqueID());
    moduleRegistry.makeExplicitModule<edm::test::sb::SBIntProducer>(desc,
                                                                    prealloc,
                                                                    &productRegistry,
                                                                    activityRegistry.preModuleConstructionSignal_,
                                                                    activityRegistry.postModuleConstructionSignal_,
                                                                    std::vector<edm::InputTag>());

    ConfigBuilder confBuilder;
    confBuilder.addModule("prod1", "edm::test::sb::SBIntProducer", EdmType::kProducer);
    confBuilder.addEndPath("e", {{"prod1"}});
    auto config = confBuilder.buildConfig();
    edm::ScheduleBuilder builder(moduleRegistry,
                                 config,
                                 confBuilder.paths(),
                                 confBuilder.endpaths(),
                                 prealloc,
                                 productRegistry,
                                 activityRegistry,
                                 processConfig);

    REQUIRE(builder.endpathNameAndModules_.size() == 1);
    REQUIRE(builder.endpathNameAndModules_.front().first == "e");
    REQUIRE(builder.endpathNameAndModules_.front().second.size() == 1);
    auto moduleInfo = builder.endpathNameAndModules_.front().second.front();
    REQUIRE(moduleInfo.description_->moduleLabel() == "prod1");
    REQUIRE(moduleInfo.description_->moduleName() == "edm::test::sb::SBIntProducer");
    REQUIRE(moduleInfo.action_ == edm::WorkerInPath::Normal);
    REQUIRE(moduleInfo.placeInPath_ == 0);
    REQUIRE(moduleInfo.runConcurrently_ == true);
    REQUIRE(builder.pathNameAndModules_.empty());
    REQUIRE(builder.allNeededModules_.size() == 1);
    REQUIRE(std::find_if(builder.allNeededModules_.begin(), builder.allNeededModules_.end(), [&desc](auto const* el) {
              return *el == desc;
            }) != builder.allNeededModules_.end());
    REQUIRE(builder.unscheduledModules_.empty());
    REQUIRE(builder.pathStatusInserters_.empty());
    REQUIRE(builder.endPathStatusInserters_.empty());
    REQUIRE(not builder.resultsInserter_);
  }
}
