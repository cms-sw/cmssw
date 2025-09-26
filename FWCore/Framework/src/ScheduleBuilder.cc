#include "FWCore/Framework/src/ScheduleBuilder.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"
#include "FWCore/Framework/src/processEDAliases.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/WorkerInPath.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

namespace edm {

  namespace {
    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    std::shared_ptr<TriggerResultInserter> makeInserter(
        ParameterSet& ioProcessPSet,
        PreallocationConfiguration const& iPrealloc,
        SignallingProductRegistryFiller& ioProductRegistry,
        ActivityRegistry& iActivityRegistry,
        std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
        ModuleRegistry& ioModuleRegistry) {
      ParameterSet* trig_pset = ioProcessPSet.getPSetForUpdate("@trigger_paths");
      trig_pset->registerIt();

      ModuleDescription md(trig_pset->id(),
                           "TriggerResultInserter",
                           "TriggerResults",
                           iProcessConfiguration.get(),
                           ModuleDescription::getUniqueID());

      return ioModuleRegistry.makeExplicitModule<TriggerResultInserter>(md,
                                                                        iPrealloc,
                                                                        &ioProductRegistry,
                                                                        iActivityRegistry.preModuleConstructionSignal_,
                                                                        iActivityRegistry.postModuleConstructionSignal_,
                                                                        *trig_pset,
                                                                        iPrealloc.numberOfStreams());
    }

    template <typename T>
    std::vector<edm::propagate_const<std::shared_ptr<T>>> makePathStatusInserters(
        std::vector<std::string> const& iPathNames,
        PreallocationConfiguration const& iPrealloc,
        SignallingProductRegistryFiller& ioProductRegistry,
        ActivityRegistry& iActivityRegistry,
        std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
        ModuleRegistry& ioModuleRegistry,
        std::string const& iModuleTypeName) {
      ParameterSet pset;
      pset.addParameter<std::string>("@module_type", iModuleTypeName);
      pset.addParameter<std::string>("@module_edm_type", "EDProducer");
      pset.registerIt();

      std::vector<edm::propagate_const<std::shared_ptr<T>>> pathStatusInserters;
      pathStatusInserters.reserve(iPathNames.size());

      for (auto const& pathName : iPathNames) {
        ModuleDescription md(
            pset.id(), iModuleTypeName, pathName, iProcessConfiguration.get(), ModuleDescription::getUniqueID());
        auto module = ioModuleRegistry.makeExplicitModule<T>(md,
                                                             iPrealloc,
                                                             &ioProductRegistry,
                                                             iActivityRegistry.preModuleConstructionSignal_,
                                                             iActivityRegistry.postModuleConstructionSignal_,
                                                             iPrealloc.numberOfStreams());
        pathStatusInserters.emplace_back(std::move(module));
      }
      return pathStatusInserters;
    }

    struct AliasInfo {
      std::string friendlyClassName;
      std::string instanceLabel;
      std::string originalInstanceLabel;
      std::string originalModuleLabel;
    };

    std::shared_ptr<edm::maker::ModuleHolder const> getModule(
        ParameterSet& ioProcessPSet,
        std::string const& iModuleLabel,
        ModuleRegistry& ioModuleRegistry,
        SignallingProductRegistryFiller& ioProductRegistry,
        ActivityRegistry& iActivityRegistry,
        PreallocationConfiguration const* iPrealloc,
        std::shared_ptr<ProcessConfiguration const> iProcessConfiguration) {
      bool isTracked = false;
      ParameterSet* modpset = ioProcessPSet.getPSetForUpdate(iModuleLabel, isTracked);
      assert(modpset != nullptr);
      assert(isTracked);

      MakeModuleParams params(modpset, ioProductRegistry, iPrealloc, iProcessConfiguration);
      return ioModuleRegistry.getModule(params,
                                        iModuleLabel,
                                        iActivityRegistry.preModuleConstructionSignal_,
                                        iActivityRegistry.postModuleConstructionSignal_);
    }

    std::vector<ModuleInPath> fillModulesInPath(ParameterSet& ioProcessPSet,
                                                ModuleRegistry& ioModuleRegistry,
                                                SignallingProductRegistryFiller& ioProductRegistry,
                                                ActivityRegistry& iActivityRegistry,
                                                PreallocationConfiguration const* iPrealloc,
                                                std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                                                std::string const& iPathName,
                                                bool iIgnoreFilters,
                                                std::vector<std::string> const& iEndPathNames) {
      auto modnames = ioProcessPSet.getParameter<std::vector<std::string>>(iPathName);
      std::vector<ModuleInPath> tmpworkers;

      unsigned int placeInPath = 0;
      for (auto const& name : modnames) {
        //Modules except EDFilters are set to run concurrently by default
        bool doNotRunConcurrently = false;
        WorkerInPath::FilterAction filterAction = WorkerInPath::Normal;
        if (name[0] == '!') {
          filterAction = WorkerInPath::Veto;
        } else if (name[0] == '-' or name[0] == '+') {
          filterAction = WorkerInPath::Ignore;
        }
        if (name[0] == '|' or name[0] == '+') {
          //cms.wait was specified so do not run concurrently
          doNotRunConcurrently = true;
        }

        std::string moduleLabel = name;
        if (filterAction != WorkerInPath::Normal or name[0] == '|') {
          moduleLabel.erase(0, 1);
        }

        auto module = getModule(ioProcessPSet,
                                moduleLabel,
                                ioModuleRegistry,
                                ioProductRegistry,
                                iActivityRegistry,
                                iPrealloc,
                                iProcessConfiguration);
        if (module == nullptr) {
          std::string pathType("endpath");
          if (std::find(iEndPathNames.begin(), iEndPathNames.end(), iPathName) == iEndPathNames.end()) {
            pathType = std::string("path");
          }
          throw Exception(errors::Configuration)
              << "The unknown module label \"" << moduleLabel << "\" appears in " << pathType << " \"" << iPathName
              << "\"\n please check spelling or remove that label from the path.";
        }

        if (iIgnoreFilters && filterAction != WorkerInPath::Ignore &&
            module->moduleType() == maker::ModuleHolder::Type::kFilter) {
          // We have a filter on an end path, and the filter is not explicitly ignored.
          // See if the filter is allowed.
          std::vector<std::string> allowed_filters =
              ioProcessPSet.getUntrackedParameter<std::vector<std::string>>("@filters_on_endpaths");
          if (std::find(allowed_filters.begin(), allowed_filters.end(), module->moduleDescription().moduleName()) ==
              allowed_filters.end()) {
            // Filter is not allowed. Ignore the result, and issue a warning.
            filterAction = WorkerInPath::Ignore;
            LogWarning("FilterOnEndPath")
                << "The EDFilter '" << module->moduleDescription().moduleName() << "' with module label '"
                << moduleLabel << "' appears on EndPath '" << iPathName << "'.\n"
                << "The return value of the filter will be ignored.\n"
                << "To suppress this warning, either remove the filter from the endpath,\n"
                << "or explicitly ignore it in the configuration by using cms.ignore().\n";
          }
        }
        bool runConcurrently = not doNotRunConcurrently;
        if (runConcurrently && module->moduleType() == maker::ModuleHolder::Type::kFilter and
            filterAction != WorkerInPath::Ignore) {
          runConcurrently = false;
        }

        tmpworkers.emplace_back(&module->moduleDescription(), filterAction, placeInPath, runConcurrently);
        ++placeInPath;
      }

      return tmpworkers;
    }

    std::vector<ModuleInPath> fillTrigPath(ParameterSet& ioProcessPSet,
                                           ModuleRegistry& ioModuleRegistry,
                                           SignallingProductRegistryFiller& ioProductRegistry,
                                           ActivityRegistry& iActivityRegistry,
                                           PreallocationConfiguration const* iPrealloc,
                                           std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                                           std::string const& iName,
                                           std::vector<std::string> const& iEndPathNames) {
      return fillModulesInPath(ioProcessPSet,
                               ioModuleRegistry,
                               ioProductRegistry,
                               iActivityRegistry,
                               iPrealloc,
                               iProcessConfiguration,
                               iName,
                               false,
                               iEndPathNames);
    }

    std::vector<ModuleInPath> fillEndPath(ParameterSet& ioProcessPSet,
                                          ModuleRegistry& ioModuleRegistry,
                                          SignallingProductRegistryFiller& ioProductRegistry,
                                          ActivityRegistry& iActivityRegistry,
                                          PreallocationConfiguration const* iPrealloc,
                                          std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                                          std::string const& iName,
                                          std::vector<std::string> const& iEndPathNames) {
      return fillModulesInPath(ioProcessPSet,
                               ioModuleRegistry,
                               ioProductRegistry,
                               iActivityRegistry,
                               iPrealloc,
                               iProcessConfiguration,
                               iName,
                               true,
                               iEndPathNames);
    }

    const std::string kFilterType("EDFilter");
    const std::string kProducerType("EDProducer");

  }  // namespace
  ScheduleBuilder::ScheduleBuilder(ModuleRegistry& iModuleRegistry,
                                   ParameterSet& ioProcessPSet,
                                   std::vector<std::string> const& iPathNames,
                                   std::vector<std::string> const& iEndPathNames,
                                   PreallocationConfiguration const& iPrealloc,
                                   SignallingProductRegistryFiller& ioProductRegistry,
                                   ActivityRegistry& iActivityRegistry,
                                   std::shared_ptr<ProcessConfiguration const> iProcessConfiguration)
      : resultsInserter_{iPathNames.empty() ? std::shared_ptr<TriggerResultInserter>{}
                                            : makeInserter(ioProcessPSet,
                                                           iPrealloc,
                                                           ioProductRegistry,
                                                           iActivityRegistry,
                                                           iProcessConfiguration,
                                                           iModuleRegistry)} {
    pathStatusInserters_ = makePathStatusInserters<PathStatusInserter>(iPathNames,
                                                                       iPrealloc,
                                                                       ioProductRegistry,
                                                                       iActivityRegistry,
                                                                       iProcessConfiguration,
                                                                       iModuleRegistry,
                                                                       std::string("PathStatusInserter"));

    if (iEndPathNames.size() > 1) {
      endPathStatusInserters_ = makePathStatusInserters<EndPathStatusInserter>(iEndPathNames,
                                                                               iPrealloc,
                                                                               ioProductRegistry,
                                                                               iActivityRegistry,
                                                                               iProcessConfiguration,
                                                                               iModuleRegistry,
                                                                               std::string("EndPathStatusInserter"));
    }

    std::set<std::string> modulesInPaths;
    //inject inserters
    iModuleRegistry.forAllModuleHolders(
        [&](auto const* holder) { modulesInPaths.insert(holder->moduleDescription().moduleLabel()); });

    pathNameAndModules_.reserve(iPathNames.size());
    for (auto const& trig_name : iPathNames) {
      pathNameAndModules_.emplace_back(trig_name,
                                       fillTrigPath(ioProcessPSet,
                                                    iModuleRegistry,
                                                    ioProductRegistry,
                                                    iActivityRegistry,
                                                    &iPrealloc,
                                                    iProcessConfiguration,
                                                    trig_name,
                                                    iEndPathNames));
      for (auto const& path : pathNameAndModules_.back().second) {
        modulesInPaths.insert(path.description_->moduleLabel());
      }
    }

    // fill normal endpaths
    endpathNameAndModules_.reserve(iEndPathNames.size());
    for (auto const& end_path_name : iEndPathNames) {
      endpathNameAndModules_.emplace_back(end_path_name,
                                          fillEndPath(ioProcessPSet,
                                                      iModuleRegistry,
                                                      ioProductRegistry,
                                                      iActivityRegistry,
                                                      &iPrealloc,
                                                      iProcessConfiguration,
                                                      end_path_name,
                                                      iEndPathNames));
      for (auto const& path : endpathNameAndModules_.back().second) {
        modulesInPaths.insert(path.description_->moduleLabel());
      }
    }

    allNeededModules_.reserve(modulesInPaths.size());
    iModuleRegistry.forAllModuleHolders([&](auto const* holder) {
      if (modulesInPaths.find(holder->moduleDescription().moduleLabel()) != modulesInPaths.end()) {
        allNeededModules_.emplace_back(&holder->moduleDescription());
      }
    });

    std::vector<std::string> modulesInConfig(ioProcessPSet.getParameter<std::vector<std::string>>("@all_modules"));
    std::set<std::string> modulesInConfigSet(modulesInConfig.begin(), modulesInConfig.end());
    std::vector<std::string> unusedLabels;
    set_difference(modulesInConfigSet.begin(),
                   modulesInConfigSet.end(),
                   modulesInPaths.begin(),
                   modulesInPaths.end(),
                   back_inserter(unusedLabels));
    std::set<std::string> unscheduledLabels;
    std::vector<std::string> shouldBeUsedLabels;
    if (!unusedLabels.empty()) {
      for (auto const& label : unusedLabels) {
        auto modType =
            ioProcessPSet.getParameter<edm::ParameterSet>(label).getParameter<std::string>("@module_edm_type");
        if (modType == kProducerType || modType == kFilterType) {
          auto module = getModule(ioProcessPSet,
                                  label,
                                  iModuleRegistry,
                                  ioProductRegistry,
                                  iActivityRegistry,
                                  &iPrealloc,
                                  iProcessConfiguration);

          unscheduledModules_.push_back(&module->moduleDescription());
          unscheduledLabels.emplace(label);
        } else {
          shouldBeUsedLabels.push_back(label);
        }
      }
      if (!shouldBeUsedLabels.empty()) {
        LogInfo("path").log([&shouldBeUsedLabels](auto& l) {
          l << "The following module labels are not assigned to any path:\n";
          l << "'" << shouldBeUsedLabels.front() << "'";
          for (std::vector<std::string>::iterator itLabel = shouldBeUsedLabels.begin() + 1,
                                                  itLabelEnd = shouldBeUsedLabels.end();
               itLabel != itLabelEnd;
               ++itLabel) {
            l << ",'" << *itLabel << "'";
          }
          l << "\n";
        });
      }
    }
    //we want the unscheduled modules at the beginning of the allNeededModules list
    allNeededModules_.insert(allNeededModules_.begin(), unscheduledModules_.begin(), unscheduledModules_.end());
  }
}  // namespace edm
