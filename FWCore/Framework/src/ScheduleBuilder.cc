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

    // If ConditionalTask modules exist in the container of module
    // names, returns the range (std::pair) for the modules. The range
    // excludes the special markers '#' (right before the
    // ConditionalTask modules) and '@' (last element).
    // If the module name container does not contain ConditionalTask
    // modules, returns std::pair of end iterators.
    template <typename T>
    auto findConditionalTaskModulesRange(T const& modnames) {
      auto beg = std::find(modnames.begin(), modnames.end(), "#");
      if (beg == modnames.end()) {
        return std::pair(modnames.end(), modnames.end());
      }
      return std::pair(beg + 1, std::prev(modnames.end()));
    }

    std::optional<std::string> findBestMatchingAlias(
        std::unordered_multimap<std::string, edm::ProductDescription const*> const& conditionalModuleBranches,
        std::unordered_multimap<std::string, AliasInfo> const& aliasMap,
        std::string const& productModuleLabel,
        ModuleConsumesInfo const& consumesInfo) {
      std::optional<std::string> best;
      int wildcardsInBest = std::numeric_limits<int>::max();
      bool bestIsAmbiguous = false;

      auto updateBest = [&best, &wildcardsInBest, &bestIsAmbiguous](
                            std::string const& label, bool instanceIsWildcard, bool typeIsWildcard) {
        int const wildcards = static_cast<int>(instanceIsWildcard) + static_cast<int>(typeIsWildcard);
        if (wildcards == 0) {
          bestIsAmbiguous = false;
          return true;
        }
        if (not best or wildcards < wildcardsInBest) {
          best = label;
          wildcardsInBest = wildcards;
          bestIsAmbiguous = false;
        } else if (best and *best != label and wildcardsInBest == wildcards) {
          bestIsAmbiguous = true;
        }
        return false;
      };

      auto findAlias = aliasMap.equal_range(productModuleLabel);
      for (auto it = findAlias.first; it != findAlias.second; ++it) {
        std::string const& aliasInstanceLabel =
            it->second.instanceLabel != "*" ? it->second.instanceLabel : it->second.originalInstanceLabel;
        bool const instanceIsWildcard = (aliasInstanceLabel == "*");
        if (instanceIsWildcard or consumesInfo.instance() == aliasInstanceLabel) {
          bool const typeIsWildcard = it->second.friendlyClassName == "*";
          if (typeIsWildcard or (consumesInfo.type().friendlyClassName() == it->second.friendlyClassName)) {
            if (updateBest(it->second.originalModuleLabel, instanceIsWildcard, typeIsWildcard)) {
              return it->second.originalModuleLabel;
            }
          } else if (consumesInfo.kindOfType() == ELEMENT_TYPE) {
            //consume is a View so need to do more intrusive search
            //find matching branches in module
            auto branches = conditionalModuleBranches.equal_range(productModuleLabel);
            for (auto itBranch = branches.first; itBranch != branches.second; ++it) {
              if (typeIsWildcard or itBranch->second->productInstanceName() == it->second.originalInstanceLabel) {
                if (productholderindexhelper::typeIsViewCompatible(consumesInfo.type(),
                                                                   TypeID(itBranch->second->wrappedType().typeInfo()),
                                                                   itBranch->second->className())) {
                  if (updateBest(it->second.originalModuleLabel, instanceIsWildcard, typeIsWildcard)) {
                    return it->second.originalModuleLabel;
                  }
                }
              }
            }
          }
        }
      }
      if (bestIsAmbiguous) {
        throw Exception(errors::UnimplementedFeature)
            << "Encountered ambiguity when trying to find a best-matching alias for\n"
            << " friendly class name " << consumesInfo.type().friendlyClassName() << "\n"
            << " module label " << productModuleLabel << "\n"
            << " product instance name " << consumesInfo.instance() << "\n"
            << "when processing EDAliases for modules in ConditionalTasks. Two aliases have the same number of "
               "wildcards ("
            << wildcardsInBest << ")";
      }
      return best;
    }

    class ConditionalTaskHelper {
    public:
      ConditionalTaskHelper(ParameterSet& ioProcessPSet,
                            SignallingProductRegistryFiller& ioProductRegistry,
                            PreallocationConfiguration const* iPrealloc,
                            std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                            ModuleRegistry& ioModuleRegistry,
                            ActivityRegistry& iActivityRegistry,
                            std::vector<std::string> const& iTrigPathNames) {
        std::unordered_set<std::string> allConditionalMods;
        for (auto const& pathName : iTrigPathNames) {
          auto const modnames = ioProcessPSet.getParameter<std::vector<std::string>>(pathName);

          //Pull out ConditionalTask modules
          auto condRange = findConditionalTaskModulesRange(modnames);
          if (condRange.first == condRange.second)
            continue;

          //the last entry should be ignored since it is required to be "@"
          allConditionalMods.insert(condRange.first, condRange.second);
        }

        for (auto const& cond : allConditionalMods) {
          //force the creation of the conditional modules so alias check can work
          // must be sure this is not added to the list of all workers else the system will hang in case the
          // module is not used.
          bool isTracked = false;
          ParameterSet* modpset = ioProcessPSet.getPSetForUpdate(cond, isTracked);
          assert(modpset != nullptr);
          assert(isTracked);

          MakeModuleParams params(modpset, ioProductRegistry, iPrealloc, iProcessConfiguration);
          (void)ioModuleRegistry.getModule(params,
                                           cond,
                                           iActivityRegistry.preModuleConstructionSignal_,
                                           iActivityRegistry.postModuleConstructionSignal_);
        }

        fillAliasMap(ioProcessPSet, allConditionalMods);
        processSwitchEDAliases(ioProcessPSet, ioProductRegistry, *iProcessConfiguration, allConditionalMods);

        //find branches created by the conditional modules
        for (auto const& prod : ioProductRegistry.registry().productList()) {
          if (allConditionalMods.find(prod.first.moduleLabel()) != allConditionalMods.end()) {
            conditionalModsBranches_.emplace(prod.first.moduleLabel(), &prod.second);
          }
        }
      }

      std::unordered_multimap<std::string, AliasInfo> const& aliasMap() const { return aliasMap_; }

      std::unordered_multimap<std::string, edm::ProductDescription const*> conditionalModuleBranches(
          std::unordered_set<std::string> const& conditionalmods) const {
        std::unordered_multimap<std::string, edm::ProductDescription const*> ret;
        for (auto const& mod : conditionalmods) {
          auto range = conditionalModsBranches_.equal_range(mod);
          ret.insert(range.first, range.second);
        }
        return ret;
      }

    private:
      void fillAliasMap(ParameterSet const& ioProcessPSet,
                        std::unordered_set<std::string> const& iAllConditionalModules) {
        auto aliases = ioProcessPSet.getParameter<std::vector<std::string>>("@all_aliases");
        std::string const star("*");
        for (auto const& alias : aliases) {
          auto info = ioProcessPSet.getParameter<edm::ParameterSet>(alias);
          auto aliasedToModuleLabels = info.getParameterNames();
          for (auto const& mod : aliasedToModuleLabels) {
            if (not mod.empty() and mod[0] != '@' and
                iAllConditionalModules.find(mod) != iAllConditionalModules.end()) {
              auto aliasVPSet = info.getParameter<std::vector<edm::ParameterSet>>(mod);
              for (auto const& aliasPSet : aliasVPSet) {
                std::string type = star;
                std::string instance = star;
                std::string originalInstance = star;
                if (aliasPSet.exists("type")) {
                  type = aliasPSet.getParameter<std::string>("type");
                }
                if (aliasPSet.exists("toProductInstance")) {
                  instance = aliasPSet.getParameter<std::string>("toProductInstance");
                }
                if (aliasPSet.exists("fromProductInstance")) {
                  originalInstance = aliasPSet.getParameter<std::string>("fromProductInstance");
                }
                aliasMap_.emplace(alias, AliasInfo{type, instance, originalInstance, mod});
              }
            }
          }
        }
      }

      void processSwitchEDAliases(ParameterSet const& ioProcessPSet,
                                  SignallingProductRegistryFiller& ioProductRegistry,
                                  ProcessConfiguration const& iProcessConfiguration,
                                  std::unordered_set<std::string> const& iAllConditionalModules) {
        auto const& all_modules = ioProcessPSet.getParameter<std::vector<std::string>>("@all_modules");
        std::vector<std::string> switchEDAliases;
        for (auto const& module : all_modules) {
          auto const& mod_pset = ioProcessPSet.getParameter<edm::ParameterSet>(module);
          if (mod_pset.getParameter<std::string>("@module_type") == "SwitchProducer") {
            auto const& all_cases = mod_pset.getParameter<std::vector<std::string>>("@all_cases");
            for (auto const& case_label : all_cases) {
              auto range = aliasMap_.equal_range(case_label);
              if (range.first != range.second) {
                switchEDAliases.push_back(case_label);
              }
            }
          }
        }
        detail::processEDAliases(switchEDAliases,
                                 iAllConditionalModules,
                                 ioProcessPSet,
                                 iProcessConfiguration.processName(),
                                 ioProductRegistry);
      }

      std::unordered_multimap<std::string, AliasInfo> aliasMap_;
      std::unordered_multimap<std::string, edm::ProductDescription const*> conditionalModsBranches_;
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

    std::vector<ModuleDescription const*> tryToPlaceConditionalModules(
        maker::ModuleHolder const* iModule,
        ModuleRegistry& iModuleRegistry,
        std::unordered_set<std::string>& ioConditionalModules,
        std::unordered_multimap<std::string, edm::ProductDescription const*> const& iConditionalModuleProducts,
        std::unordered_multimap<std::string, AliasInfo> const& iAliasMap,
        ParameterSet& ioProcessPSet,
        SignallingProductRegistryFiller& ioProductRegistry,
        ActivityRegistry& iActivityRegistry,
        PreallocationConfiguration const* iPrealloc,
        std::shared_ptr<ProcessConfiguration const> iProcessConfiguration) {
      std::vector<ModuleDescription const*> returnValue;
      //auto const& consumesInfo = worker->moduleConsumesInfos();
      auto const& consumesInfo = iModule->moduleConsumesInfos();
      using namespace productholderindexhelper;
      for (auto const& ci : consumesInfo) {
        if (not ci.skipCurrentProcess() and
            (ci.process().empty() or ci.process() == iProcessConfiguration->processName())) {
          auto productModuleLabel = std::string(ci.label());
          bool productFromConditionalModule = false;
          auto itFound = ioConditionalModules.find(productModuleLabel);
          if (itFound == ioConditionalModules.end()) {
            //Check to see if this was an alias
            //note that aliasMap was previously filtered so only the conditional modules remain there
            auto foundAlias = findBestMatchingAlias(iConditionalModuleProducts, iAliasMap, productModuleLabel, ci);
            if (foundAlias) {
              productModuleLabel = *foundAlias;
              productFromConditionalModule = true;
              itFound = ioConditionalModules.find(productModuleLabel);
              //check that the alias-for conditional module has not been used
              if (itFound == ioConditionalModules.end()) {
                continue;
              }
            }
          } else {
            //need to check the rest of the data product info
            auto findBranches = iConditionalModuleProducts.equal_range(productModuleLabel);
            for (auto itBranch = findBranches.first; itBranch != findBranches.second; ++itBranch) {
              if (itBranch->second->productInstanceName() == ci.instance()) {
                if (ci.kindOfType() == PRODUCT_TYPE) {
                  if (ci.type() == itBranch->second->unwrappedTypeID()) {
                    productFromConditionalModule = true;
                    break;
                  }
                } else {
                  //this is a view
                  if (typeIsViewCompatible(ci.type(),
                                           TypeID(itBranch->second->wrappedType().typeInfo()),
                                           itBranch->second->className())) {
                    productFromConditionalModule = true;
                    break;
                  }
                }
              }
            }
          }
          if (productFromConditionalModule) {
            auto condModule = getModule(ioProcessPSet,
                                        productModuleLabel,
                                        iModuleRegistry,
                                        ioProductRegistry,
                                        iActivityRegistry,
                                        iPrealloc,
                                        iProcessConfiguration);
            assert(condModule);

            ioConditionalModules.erase(itFound);

            auto dependents = tryToPlaceConditionalModules(condModule.get(),
                                                           iModuleRegistry,
                                                           ioConditionalModules,
                                                           iConditionalModuleProducts,
                                                           iAliasMap,
                                                           ioProcessPSet,
                                                           ioProductRegistry,
                                                           iActivityRegistry,
                                                           iPrealloc,
                                                           iProcessConfiguration);
            returnValue.insert(returnValue.end(), dependents.begin(), dependents.end());
            returnValue.emplace_back(&condModule->moduleDescription());
          }
        }
      }
      return returnValue;
    }

    std::vector<ModuleInPath> fillModulesInPath(ParameterSet& ioProcessPSet,
                                                ModuleRegistry& ioModuleRegistry,
                                                SignallingProductRegistryFiller& ioProductRegistry,
                                                ActivityRegistry& iActivityRegistry,
                                                PreallocationConfiguration const* iPrealloc,
                                                std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                                                std::string const& iPathName,
                                                bool iIgnoreFilters,
                                                std::vector<std::string> const& iEndPathNames,
                                                ConditionalTaskHelper const& iConditionalTaskHelper,
                                                std::unordered_set<std::string>& oAllConditionalModules) {
      auto modnames = ioProcessPSet.getParameter<std::vector<std::string>>(iPathName);
      std::vector<ModuleInPath> tmpworkers;

      //Pull out ConditionalTask modules
      auto condRange = findConditionalTaskModulesRange(modnames);

      std::unordered_set<std::string> conditionalmods;
      //An EDAlias may be redirecting to a module on a ConditionalTask
      std::unordered_multimap<std::string, edm::ProductDescription const*> conditionalModsBranches;
      std::unordered_map<std::string, unsigned int> conditionalModOrder;
      if (condRange.first != condRange.second) {
        for (auto it = condRange.first; it != condRange.second; ++it) {
          // ordering needs to skip the # token in the path list
          conditionalModOrder.emplace(*it, it - modnames.begin() - 1);
        }
        //the last entry should be ignored since it is required to be "@"
        conditionalmods = std::unordered_set<std::string>(std::make_move_iterator(condRange.first),
                                                          std::make_move_iterator(condRange.second));

        conditionalModsBranches = iConditionalTaskHelper.conditionalModuleBranches(conditionalmods);
        modnames.erase(std::prev(condRange.first), modnames.end());

        // Make a union of all conditional modules from all Paths
        oAllConditionalModules.insert(conditionalmods.begin(), conditionalmods.end());
      }

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

        auto condModules = tryToPlaceConditionalModules(module.get(),
                                                        ioModuleRegistry,
                                                        conditionalmods,
                                                        conditionalModsBranches,
                                                        iConditionalTaskHelper.aliasMap(),
                                                        ioProcessPSet,
                                                        ioProductRegistry,
                                                        iActivityRegistry,
                                                        iPrealloc,
                                                        iProcessConfiguration);
        for (auto condMod : condModules) {
          tmpworkers.emplace_back(condMod, WorkerInPath::Ignore, conditionalModOrder[condMod->moduleLabel()], true);
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
                                           std::vector<std::string> const& iEndPathNames,
                                           ConditionalTaskHelper const& iConditionalTaskHelper,
                                           std::unordered_set<std::string>& oAllConditionalModules) {
      return fillModulesInPath(ioProcessPSet,
                               ioModuleRegistry,
                               ioProductRegistry,
                               iActivityRegistry,
                               iPrealloc,
                               iProcessConfiguration,
                               iName,
                               false,
                               iEndPathNames,
                               iConditionalTaskHelper,
                               oAllConditionalModules);
    }

    std::vector<ModuleInPath> fillEndPath(ParameterSet& ioProcessPSet,
                                          ModuleRegistry& ioModuleRegistry,
                                          SignallingProductRegistryFiller& ioProductRegistry,
                                          ActivityRegistry& iActivityRegistry,
                                          PreallocationConfiguration const* iPrealloc,
                                          std::shared_ptr<ProcessConfiguration const> iProcessConfiguration,
                                          std::string const& iName,
                                          std::vector<std::string> const& iEndPathNames,
                                          ConditionalTaskHelper const& iConditionalTaskHelper,
                                          std::unordered_set<std::string>& oAllConditionalModules) {
      return fillModulesInPath(ioProcessPSet,
                               ioModuleRegistry,
                               ioProductRegistry,
                               iActivityRegistry,
                               iPrealloc,
                               iProcessConfiguration,
                               iName,
                               true,
                               iEndPathNames,
                               iConditionalTaskHelper,
                               oAllConditionalModules);
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

    ConditionalTaskHelper conditionalTaskHelper(ioProcessPSet,
                                                ioProductRegistry,
                                                &iPrealloc,
                                                iProcessConfiguration,
                                                iModuleRegistry,
                                                iActivityRegistry,
                                                iPathNames);
    std::unordered_set<std::string> conditionalModules;

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
                                                    iEndPathNames,
                                                    conditionalTaskHelper,
                                                    conditionalModules));
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
                                                      iEndPathNames,
                                                      conditionalTaskHelper,
                                                      conditionalModules));
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