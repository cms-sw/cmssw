#include "FWCore/Framework/interface/Schedule.h"

#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/interface/TriggerTimingReport.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/ModuleRegistry.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/TypeID.h"



#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <list>
#include <map>
#include <set>
#include <exception>
#include <sstream>

#include "make_shared_noexcept_false.h"


namespace edm {

  class Maker;

  namespace {
    using std::placeholders::_1;

    bool binary_search_string(std::vector<std::string> const& v, std::string const& s) {
      return std::binary_search(v.begin(), v.end(), s);
    }

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    std::shared_ptr<TriggerResultInserter>
    makeInserter(ParameterSet& proc_pset,
                 PreallocationConfiguration const& iPrealloc,
                 ProductRegistry& preg,
                 ExceptionToActionTable const& actions,
                 std::shared_ptr<ActivityRegistry> areg,
                 std::shared_ptr<ProcessConfiguration> processConfiguration) {

      ParameterSet* trig_pset = proc_pset.getPSetForUpdate("@trigger_paths");
      trig_pset->registerIt();

      WorkerParams work_args(trig_pset, preg, &iPrealloc, processConfiguration, actions);
      ModuleDescription md(trig_pset->id(),
                           "TriggerResultInserter",
                           "TriggerResults",
                           processConfiguration.get(),
                           ModuleDescription::getUniqueID());

      areg->preModuleConstructionSignal_(md);
      bool postCalled = false;
      std::shared_ptr<TriggerResultInserter> returnValue;
      try {
        maker::ModuleHolderT<TriggerResultInserter> holder(make_shared_noexcept_false<TriggerResultInserter>(*trig_pset, iPrealloc.numberOfStreams()),static_cast<Maker const*>(nullptr));
        holder.setModuleDescription(md);
        holder.registerProductsAndCallbacks(&preg);
        returnValue =holder.module();
        postCalled = true;
        // if exception then post will be called in the catch block
        areg->postModuleConstructionSignal_(md);
      }
      catch (...) {
        if(!postCalled) {
          try {
            areg->postModuleConstructionSignal_(md);
          }
          catch (...) {
            // If post throws an exception ignore it because we are already handling another exception
          }
        }
        throw;
      }
      return returnValue;
    }

    template <typename T>
    void
    makePathStatusInserters(std::vector<edm::propagate_const<std::shared_ptr<T>>>& pathStatusInserters,
                            std::vector<std::string> const& pathNames,
                            PreallocationConfiguration const& iPrealloc,
                            ProductRegistry& preg,
                            std::shared_ptr<ActivityRegistry> areg,
                            std::shared_ptr<ProcessConfiguration> processConfiguration,
                            std::string const& moduleTypeName) {

      ParameterSet pset;
      pset.addParameter<std::string>("@module_type", moduleTypeName);
      pset.addParameter<std::string>("@module_edm_type", "EDProducer");
      pset.registerIt();

      pathStatusInserters.reserve(pathNames.size());

      for (auto const& pathName : pathNames) {

        ModuleDescription md(pset.id(),
                             moduleTypeName,
                             pathName,
                             processConfiguration.get(),
                             ModuleDescription::getUniqueID());

        areg->preModuleConstructionSignal_(md);
        bool postCalled = false;

        try {
          maker::ModuleHolderT<T> holder(make_shared_noexcept_false<T>(iPrealloc.numberOfStreams()),
                                         static_cast<Maker const*>(nullptr));
          holder.setModuleDescription(md);
          holder.registerProductsAndCallbacks(&preg);
          pathStatusInserters.emplace_back(holder.module());
          postCalled = true;
          // if exception then post will be called in the catch block
          areg->postModuleConstructionSignal_(md);
        }
        catch (...) {
          if(!postCalled) {
            try {
              areg->postModuleConstructionSignal_(md);
            }
            catch (...) {
              // If post throws an exception ignore it because we are already handling another exception
            }
          }
          throw;
        }
      }
    }

    void
    checkAndInsertAlias(std::string const& friendlyClassName,
                        std::string const& moduleLabel,
                        std::string const& productInstanceName,
                        std::string const& processName,
                        std::string const& alias,
                        std::string const& instanceAlias,
                        ProductRegistry const& preg,
                        std::multimap<BranchKey, BranchKey>& aliasMap,
                        std::map<BranchKey, BranchKey>& aliasKeys) {
      std::string const star("*");

      BranchKey key(friendlyClassName, moduleLabel, productInstanceName, processName);
      if(preg.productList().find(key) == preg.productList().end()) {
        // No product was found matching the alias.
        // We throw an exception only if a module with the specified module label was created in this process.
        for(auto const& product : preg.productList()) {
          if(moduleLabel == product.first.moduleLabel() && processName == product.first.processName()) {
            throw Exception(errors::Configuration, "EDAlias does not match data\n")
              << "There are no products of type '" << friendlyClassName << "'\n"
              << "with module label '" << moduleLabel << "' and instance name '" << productInstanceName << "'.\n";
          }
        }
      }

      std::string const& theInstanceAlias(instanceAlias == star ? productInstanceName : instanceAlias);
      BranchKey aliasKey(friendlyClassName, alias, theInstanceAlias, processName);
      if(preg.productList().find(aliasKey) != preg.productList().end()) {
        throw Exception(errors::Configuration, "EDAlias conflicts with data\n")
          << "A product of type '" << friendlyClassName << "'\n"
          << "with module label '" << alias << "' and instance name '" << theInstanceAlias << "'\n"
          << "already exists.\n";
      }
      auto iter = aliasKeys.find(aliasKey);
      if(iter != aliasKeys.end()) {
        // The alias matches a previous one.  If the same alias is used for different product, throw.
        if(iter->second != key) {
          throw Exception(errors::Configuration, "EDAlias conflict\n")
            << "The module label alias '" << alias << "' and product instance alias '" << theInstanceAlias << "'\n"
            << "are used for multiple products of type '" << friendlyClassName << "'\n"
            << "One has module label '" << moduleLabel << "' and product instance name '" << productInstanceName << "',\n"
            << "the other has module label '" << iter->second.moduleLabel() << "' and product instance name '" << iter->second.productInstanceName() << "'.\n";
        }
      } else {
        auto prodIter = preg.productList().find(key);
        if(prodIter != preg.productList().end()) {
          if (!prodIter->second.produced()) {
            throw Exception(errors::Configuration, "EDAlias\n")
              << "The module label alias '" << alias << "' and product instance alias '" << theInstanceAlias << "'\n"
              << "are used for a product of type '" << friendlyClassName << "'\n"
              << "with module label '" << moduleLabel << "' and product instance name '" << productInstanceName << "',\n"
              << "An EDAlias can only be used for products produced in the current process. This one is not.\n";
          }
          aliasMap.insert(std::make_pair(key, aliasKey));
          aliasKeys.insert(std::make_pair(aliasKey, key));
        }
      }
    }

    void
    processEDAliases(ParameterSet const& proc_pset, std::string const& processName, ProductRegistry& preg) {
      std::vector<std::string> aliases = proc_pset.getParameter<std::vector<std::string> >("@all_aliases");
      if(aliases.empty()) {
        return;
      }
      std::string const star("*");
      std::string const empty("");
      ParameterSetDescription desc;
      desc.add<std::string>("type");
      desc.add<std::string>("fromProductInstance", star);
      desc.add<std::string>("toProductInstance", star);

      std::multimap<BranchKey, BranchKey> aliasMap;

      std::map<BranchKey, BranchKey> aliasKeys; // Used to search for duplicates or clashes.

      // Now, loop over the alias information and store it in aliasMap.
      for(std::string const& alias : aliases) {
        ParameterSet const& aliasPSet = proc_pset.getParameterSet(alias);
        std::vector<std::string> vPSetNames = aliasPSet.getParameterNamesForType<VParameterSet>();
        for(std::string const& moduleLabel : vPSetNames) {
          VParameterSet vPSet = aliasPSet.getParameter<VParameterSet>(moduleLabel);
          for(ParameterSet& pset : vPSet) {
            desc.validate(pset);
            std::string friendlyClassName = pset.getParameter<std::string>("type");
            std::string productInstanceName = pset.getParameter<std::string>("fromProductInstance");
            std::string instanceAlias = pset.getParameter<std::string>("toProductInstance");
            if(productInstanceName == star) {
              bool match = false;
              BranchKey lowerBound(friendlyClassName, moduleLabel, empty, empty);
              for(ProductRegistry::ProductList::const_iterator it = preg.productList().lower_bound(lowerBound);
                  it != preg.productList().end() && it->first.friendlyClassName() == friendlyClassName && it->first.moduleLabel() == moduleLabel;
                  ++it) {
                if(it->first.processName() != processName) {
                  continue;
                }
                match = true;

                checkAndInsertAlias(friendlyClassName, moduleLabel, it->first.productInstanceName(), processName, alias, instanceAlias, preg, aliasMap, aliasKeys);
              }
              if(!match) {
                // No product was found matching the alias.
                // We throw an exception only if a module with the specified module label was created in this process.
                for(auto const& product : preg.productList()) {
                  if(moduleLabel == product.first.moduleLabel() && processName == product.first.processName()) {
                    throw Exception(errors::Configuration, "EDAlias parameter set mismatch\n")
                       << "There are no products of type '" << friendlyClassName << "'\n"
                       << "with module label '" << moduleLabel << "'.\n";
                  }
                }
              }
            } else {
              checkAndInsertAlias(friendlyClassName, moduleLabel, productInstanceName, processName, alias, instanceAlias, preg, aliasMap, aliasKeys);
            }
          }
        }
      }


      // Now add the new alias entries to the product registry.
      for(auto const& aliasEntry : aliasMap) {
        ProductRegistry::ProductList::const_iterator it = preg.productList().find(aliasEntry.first);
        assert(it != preg.productList().end());
        preg.addLabelAlias(it->second, aliasEntry.second.moduleLabel(), aliasEntry.second.productInstanceName());
      }

    }

    typedef std::vector<std::string> vstring;

    void processSwitchProducers(ParameterSet const& proc_pset, std::string const& processName, ProductRegistry& preg) {
      // Update Switch BranchDescriptions for the chosen case
      std::vector<BranchKey> chosenBranches;
      std::map<std::string, std::vector<std::string> > allCasesMap;
      for(auto& prod: preg.productListUpdator()) {
        if(prod.second.isSwitchAlias()) {
          if(allCasesMap.find(prod.second.moduleLabel()) == allCasesMap.end()) {
            auto const& switchPSet = proc_pset.getParameter<edm::ParameterSet>(prod.second.moduleLabel());
            allCasesMap[prod.second.moduleLabel()] = switchPSet.getParameter<std::vector<std::string>>("@all_cases");
          }

          for(auto const& item: preg.productList()) {
            if(item.second.branchType() == prod.second.branchType() and
               item.second.unwrappedTypeID().typeInfo() == prod.second.unwrappedTypeID().typeInfo() and
               item.first.moduleLabel() == prod.second.switchAliasModuleLabel() and
               item.first.productInstanceName() == prod.second.productInstanceName()
               ) {
              if(item.first.processName() != processName) {
                throw Exception(errors::LogicError)
                  << "Encountered a BranchDescription that is aliased-for by SwitchProducer, and whose processName " << item.first.processName() << " differs from current process " << processName
                  << ". Module label is " << item.first.moduleLabel() << ".\nPlease contact a framework developer.";
              }
              prod.second.setSwitchAliasForBranch(item.second);
              chosenBranches.push_back(prod.first); // with moduleLabel of the Switch
            }
          }
        }
      }
      if(allCasesMap.empty())
        return;

      std::sort(chosenBranches.begin(), chosenBranches.end());

      // Check that non-chosen cases declare exactly the same branches
      auto foundBranches = std::vector<bool>(chosenBranches.size(), false);
      for(auto const& switchItem: allCasesMap) {
        auto const& switchLabel = switchItem.first;
        auto const& caseLabels = switchItem.second;
        for(auto const& caseLabel: caseLabels) {
          std::fill(foundBranches.begin(), foundBranches.end(), false);
          for(auto const& item: preg.productList()) {
            if(item.first.moduleLabel() == caseLabel) {
              auto range = std::equal_range(chosenBranches.begin(), chosenBranches.end(), BranchKey(item.first.friendlyClassName(),
                                                                                                    switchLabel,
                                                                                                    item.first.productInstanceName(),
                                                                                                    item.first.processName()));
              if(range.first == range.second) {
                throw Exception(errors::Configuration)
                  << "SwitchProducer " << switchLabel << " has a case " << caseLabel << " with a product " << item.first << " that is not produced by the chosen case " << proc_pset.getParameter<edm::ParameterSet>(switchLabel).getUntrackedParameter<std::string>("@chosen_case");
              }
              assert(std::distance(range.first, range.second) == 1);
              foundBranches[std::distance(chosenBranches.begin(), range.first)] = true;

              // Check that there are no BranchAliases for any of the cases
              auto const& bd = item.second;
              if(not bd.branchAliases().empty()) {
                auto ex = Exception(errors::UnimplementedFeature) << "SwitchProducer does not support ROOT branch aliases. Got the following ROOT branch aliases for SwitchProducer with label " << switchLabel << " for case " << caseLabel << ":";
                for(auto const& item: bd.branchAliases()) {
                  ex << " " << item;
                }
                throw ex;
              }
            }
          }

          for(size_t i=0; i<chosenBranches.size(); i++) {
            if(not foundBranches[i]) {
              throw Exception(errors::Configuration)
                << "SwitchProducer " << switchLabel << " has a case " << caseLabel << " that does not produce a product " << chosenBranches[i] << " that is produced by the chosen case " << proc_pset.getParameter<edm::ParameterSet>(switchLabel).getUntrackedParameter<std::string>("@chosen_case");
            }
          }
        }
      }
    }

    void reduceParameterSet(ParameterSet& proc_pset,
                            vstring const& end_path_name_list,
                            vstring& modulesInConfig,
                            std::set<std::string> const& usedModuleLabels,
                            std::map<std::string, std::vector<std::pair<std::string, int> > >& outputModulePathPositions) {
      // Before calculating the ParameterSetID of the top level ParameterSet or
      // saving it in the registry drop from the top level ParameterSet all
      // OutputModules and EDAnalyzers not on trigger paths. If unscheduled
      // production is not enabled also drop all the EDFilters and EDProducers
      // that are not scheduled. Drop the ParameterSet used to configure the module
      // itself. Also drop the other traces of these labels in the top level
      // ParameterSet: Remove that labels from @all_modules and from all the
      // end paths. If this makes any end paths empty, then remove the end path
      // name from @end_paths, and @paths.

      // First make a list of labels to drop
      vstring outputModuleLabels;
      std::string edmType;
      std::string const moduleEdmType("@module_edm_type");
      std::string const outputModule("OutputModule");
      std::string const edAnalyzer("EDAnalyzer");
      std::string const edFilter("EDFilter");
      std::string const edProducer("EDProducer");

      std::set<std::string> modulesInConfigSet(modulesInConfig.begin(), modulesInConfig.end());

      //need a list of all modules on paths in order to determine
      // if an EDAnalyzer only appears on an end path
      vstring scheduledPaths = proc_pset.getParameter<vstring>("@paths");
      std::set<std::string> modulesOnPaths;
      {
        std::set<std::string> noEndPaths(scheduledPaths.begin(),scheduledPaths.end());
        for(auto const& endPath: end_path_name_list) {
          noEndPaths.erase(endPath);
        }
        {
          vstring labels;
          for(auto const& path: noEndPaths) {
            labels = proc_pset.getParameter<vstring>(path);
            modulesOnPaths.insert(labels.begin(),labels.end());
          }
        }
      }
      //Initially fill labelsToBeDropped with all module mentioned in
      // the configuration but which are not being used by the system
      std::vector<std::string> labelsToBeDropped;
      labelsToBeDropped.reserve(modulesInConfigSet.size());
      std::set_difference(modulesInConfigSet.begin(),modulesInConfigSet.end(),
                          usedModuleLabels.begin(),usedModuleLabels.end(),
                          std::back_inserter(labelsToBeDropped));

      const unsigned int sizeBeforeOutputModules = labelsToBeDropped.size();
      for (auto const& modLabel: usedModuleLabels) {
        // Do nothing for modules that do not have a ParameterSet. Modules of type
        // PathStatusInserter and EndPathStatusInserter will not have a ParameterSet.
        if (proc_pset.existsAs<ParameterSet>(modLabel)) {
          edmType = proc_pset.getParameterSet(modLabel).getParameter<std::string>(moduleEdmType);
          if (edmType == outputModule) {
            outputModuleLabels.push_back(modLabel);
            labelsToBeDropped.push_back(modLabel);
          }
          if(edmType == edAnalyzer) {
            if(modulesOnPaths.end()==modulesOnPaths.find(modLabel)) {
              labelsToBeDropped.push_back(modLabel);
            }
          }
        }
      }
      //labelsToBeDropped must be sorted
      std::inplace_merge(labelsToBeDropped.begin(),
                         labelsToBeDropped.begin()+sizeBeforeOutputModules,
                         labelsToBeDropped.end());

      // drop the parameter sets used to configure the modules
      for_all(labelsToBeDropped, std::bind(&ParameterSet::eraseOrSetUntrackedParameterSet, std::ref(proc_pset), _1));

      // drop the labels from @all_modules
      vstring::iterator endAfterRemove = std::remove_if(modulesInConfig.begin(), modulesInConfig.end(), std::bind(binary_search_string, std::ref(labelsToBeDropped), _1));
      modulesInConfig.erase(endAfterRemove, modulesInConfig.end());
      proc_pset.addParameter<vstring>(std::string("@all_modules"), modulesInConfig);

      // drop the labels from all end paths
      vstring endPathsToBeDropped;
      vstring labels;
      for (vstring::const_iterator iEndPath = end_path_name_list.begin(), endEndPath = end_path_name_list.end();
           iEndPath != endEndPath;
           ++iEndPath) {
        labels = proc_pset.getParameter<vstring>(*iEndPath);
        vstring::iterator iSave = labels.begin();
        vstring::iterator iBegin = labels.begin();

        for (vstring::iterator iLabel = labels.begin(), iEnd = labels.end();
             iLabel != iEnd; ++iLabel) {
          if (binary_search_string(labelsToBeDropped, *iLabel)) {
            if (binary_search_string(outputModuleLabels, *iLabel)) {
              outputModulePathPositions[*iLabel].emplace_back(*iEndPath, iSave - iBegin);
            }
          } else {
            if (iSave != iLabel) {
              iSave->swap(*iLabel);
            }
            ++iSave;
          }
        }
        labels.erase(iSave, labels.end());
        if (labels.empty()) {
          // remove empty end paths and save their names
          proc_pset.eraseSimpleParameter(*iEndPath);
          endPathsToBeDropped.push_back(*iEndPath);
        } else {
          proc_pset.addParameter<vstring>(*iEndPath, labels);
        }
      }
      sort_all(endPathsToBeDropped);

      // remove empty end paths from @paths
      endAfterRemove = std::remove_if(scheduledPaths.begin(), scheduledPaths.end(), std::bind(binary_search_string, std::ref(endPathsToBeDropped), _1));
      scheduledPaths.erase(endAfterRemove, scheduledPaths.end());
      proc_pset.addParameter<vstring>(std::string("@paths"), scheduledPaths);

      // remove empty end paths from @end_paths
      vstring scheduledEndPaths = proc_pset.getParameter<vstring>("@end_paths");
      endAfterRemove = std::remove_if(scheduledEndPaths.begin(), scheduledEndPaths.end(), std::bind(binary_search_string, std::ref(endPathsToBeDropped), _1));
      scheduledEndPaths.erase(endAfterRemove, scheduledEndPaths.end());
      proc_pset.addParameter<vstring>(std::string("@end_paths"), scheduledEndPaths);

    }

    class RngEDConsumer : public EDConsumerBase {
    public:
      explicit RngEDConsumer(std::set<TypeID>& typesConsumed) {
        Service<RandomNumberGenerator> rng;
        if(rng.isAvailable()) {
          rng->consumes(consumesCollector());
          for (auto const& consumesInfo : this->consumesInfo()) {
            typesConsumed.emplace(consumesInfo.type());
          }
        }
      }
    };
  }
  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  Schedule::Schedule(ParameterSet& proc_pset,
                     service::TriggerNamesService const& tns,
                     ProductRegistry& preg,
                     BranchIDListHelper& branchIDListHelper,
                     ThinnedAssociationsHelper& thinnedAssociationsHelper,
                     SubProcessParentageHelper const* subProcessParentageHelper,
                     ExceptionToActionTable const& actions,
                     std::shared_ptr<ActivityRegistry> areg,
                     std::shared_ptr<ProcessConfiguration> processConfiguration,
                     bool hasSubprocesses,
                     PreallocationConfiguration const& prealloc,
                     ProcessContext const* processContext) :
  //Only create a resultsInserter if there is a trigger path
  resultsInserter_{tns.getTrigPaths().empty()? std::shared_ptr<TriggerResultInserter>{} :makeInserter(proc_pset,prealloc,preg,actions,areg,processConfiguration)},
    moduleRegistry_(new ModuleRegistry()),
    all_output_communicators_(),
    preallocConfig_(prealloc),
    pathNames_(&tns.getTrigPaths()),
    endPathNames_(&tns.getEndPaths()),
    wantSummary_(tns.wantSummary()),
    endpathsAreActive_(true)
  {
    makePathStatusInserters(pathStatusInserters_,
                            *pathNames_,
                            prealloc,
                            preg,
                            areg,
                            processConfiguration,
                            std::string("PathStatusInserter"));

    makePathStatusInserters(endPathStatusInserters_,
                            *endPathNames_,
                            prealloc,
                            preg,
                            areg,
                            processConfiguration,
                            std::string("EndPathStatusInserter"));

    assert(0<prealloc.numberOfStreams());
    streamSchedules_.reserve(prealloc.numberOfStreams());
    for(unsigned int i=0; i<prealloc.numberOfStreams();++i) {
      streamSchedules_.emplace_back(make_shared_noexcept_false<StreamSchedule>(
        resultsInserter(),
        pathStatusInserters_,
        endPathStatusInserters_,
        moduleRegistry(),
        proc_pset,tns,prealloc,preg,
        branchIDListHelper,actions,
        areg,processConfiguration,
        !hasSubprocesses,
        StreamID{i},
        processContext));
    }

    //TriggerResults are injected automatically by StreamSchedules and are
    // unknown to the ModuleRegistry
    const std::string kTriggerResults("TriggerResults");
    std::vector<std::string> modulesToUse;
    modulesToUse.reserve(streamSchedules_[0]->allWorkers().size());
    for(auto const& worker : streamSchedules_[0]->allWorkers()) {
      if(worker->description().moduleLabel() != kTriggerResults) {
        modulesToUse.push_back(worker->description().moduleLabel());
      }
    }
    //The unscheduled modules are at the end of the list, but we want them at the front
    unsigned int n = streamSchedules_[0]->numberOfUnscheduledModules();
    if(n>0) {
      std::vector<std::string> temp;
      temp.reserve(modulesToUse.size());
      auto itBeginUnscheduled = modulesToUse.begin()+modulesToUse.size()-n;
      std::copy(itBeginUnscheduled,modulesToUse.end(),
                std::back_inserter(temp));
      std::copy(modulesToUse.begin(),itBeginUnscheduled,std::back_inserter(temp));
      temp.swap(modulesToUse);
    }

    // propagate_const<T> has no reset() function
    globalSchedule_ = std::make_unique<GlobalSchedule>(
      resultsInserter(),
      pathStatusInserters_,
      endPathStatusInserters_,
      moduleRegistry(),
      modulesToUse,
      proc_pset, preg, prealloc,
      actions,areg,processConfiguration,processContext);

    //TriggerResults is not in the top level ParameterSet so the call to
    // reduceParameterSet would fail to find it. Just remove it up front.
    std::set<std::string> usedModuleLabels;
    for(auto const& worker: allWorkers()) {
      if(worker->description().moduleLabel() != kTriggerResults) {
        usedModuleLabels.insert(worker->description().moduleLabel());
      }
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string> >("@all_modules"));
    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    reduceParameterSet(proc_pset, tns.getEndPaths(), modulesInConfig, usedModuleLabels,
                       outputModulePathPositions);
    processEDAliases(proc_pset, processConfiguration->processName(), preg);
    processSwitchProducers(proc_pset, processConfiguration->processName(), preg);
    proc_pset.registerIt();
    processConfiguration->setParameterSetID(proc_pset.id());
    processConfiguration->setProcessConfigurationID();

    // This is used for a little sanity-check to make sure no code
    // modifications alter the number of workers at a later date.
    size_t all_workers_count = allWorkers().size();

    moduleRegistry_->forAllModuleHolders([this](maker::ModuleHolder* iHolder){
      auto comm = iHolder->createOutputModuleCommunicator();
      if (comm) {
        all_output_communicators_.emplace_back(std::shared_ptr<OutputModuleCommunicator>{comm.release()});
      }
    });
    // Now that the output workers are filled in, set any output limits or information.
    limitOutput(proc_pset, branchIDListHelper.branchIDLists(), subProcessParentageHelper);

    // Sanity check: make sure nobody has added a worker after we've
    // already relied on the WorkerManager being full.
    assert (all_workers_count == allWorkers().size());

    branchIDListHelper.updateFromRegistry(preg);

    for(auto const& worker : streamSchedules_[0]->allWorkers()) {
      worker->registerThinnedAssociations(preg, thinnedAssociationsHelper);
    }
    thinnedAssociationsHelper.sort();

    // The output modules consume products in kept branches.
    // So we must set this up before freezing.
    for (auto& c : all_output_communicators_) {
      c->selectProducts(preg, thinnedAssociationsHelper);
    }

    for(auto & product : preg.productListUpdator()) {
      setIsMergeable(product.second);
    }

    {
      // We now get a collection of types that may be consumed.
      std::set<TypeID> productTypesConsumed;
      std::set<TypeID> elementTypesConsumed;
      // Loop over all modules
      for (auto const& worker : allWorkers()) {
        for (auto const& consumesInfo : worker->consumesInfo()) {
          if (consumesInfo.kindOfType() == PRODUCT_TYPE) {
            productTypesConsumed.emplace(consumesInfo.type());
          } else {
            elementTypesConsumed.emplace(consumesInfo.type());
          }
        }
      }
      // The SubProcess class is not a module, yet it may consume.
      if(hasSubprocesses) {
        productTypesConsumed.emplace(typeid(TriggerResults));
      }
      // The RandomNumberGeneratorService is not a module, yet it consumes.
      {
         RngEDConsumer rngConsumer = RngEDConsumer(productTypesConsumed);
      }
      preg.setFrozen(productTypesConsumed, elementTypesConsumed, processConfiguration->processName());
    }

    for (auto& c : all_output_communicators_) {
      c->setEventSelectionInfo(outputModulePathPositions, preg.anyProductProduced());
    }

    if(wantSummary_) {
      std::vector<const ModuleDescription*> modDesc;
      const auto& workers = allWorkers();
      modDesc.reserve(workers.size());

      std::transform(workers.begin(),workers.end(),
                     std::back_inserter(modDesc),
                     [](const Worker* iWorker) -> const ModuleDescription* {
                       return iWorker->descPtr();
                     });

      // propagate_const<T> has no reset() function
      summaryTimeKeeper_ = std::make_unique<SystemTimeKeeper>(
                                                    prealloc.numberOfStreams(),
                                                    modDesc,
                                                    tns,
                                                    processContext);
      auto timeKeeperPtr = summaryTimeKeeper_.get();

      areg->watchPreModuleEvent(timeKeeperPtr, &SystemTimeKeeper::startModuleEvent);
      areg->watchPostModuleEvent(timeKeeperPtr, &SystemTimeKeeper::stopModuleEvent);
      areg->watchPreModuleEventAcquire(timeKeeperPtr, &SystemTimeKeeper::restartModuleEvent);
      areg->watchPostModuleEventAcquire(timeKeeperPtr, &SystemTimeKeeper::stopModuleEvent);
      areg->watchPreModuleEventDelayedGet(timeKeeperPtr, &SystemTimeKeeper::pauseModuleEvent);
      areg->watchPostModuleEventDelayedGet(timeKeeperPtr,&SystemTimeKeeper::restartModuleEvent);

      areg->watchPreSourceEvent(timeKeeperPtr, &SystemTimeKeeper::startEvent);
      areg->watchPostEvent(timeKeeperPtr, &SystemTimeKeeper::stopEvent);

      areg->watchPrePathEvent(timeKeeperPtr, &SystemTimeKeeper::startPath);
      areg->watchPostPathEvent(timeKeeperPtr, &SystemTimeKeeper::stopPath);

      areg->watchPostBeginJob(timeKeeperPtr, &SystemTimeKeeper::startProcessingLoop);
      areg->watchPreEndJob(timeKeeperPtr, &SystemTimeKeeper::stopProcessingLoop);
      //areg->preModuleEventSignal_.connect([timeKeeperPtr](StreamContext const& iContext, ModuleCallingContext const& iMod) {
      //timeKeeperPtr->startModuleEvent(iContext,iMod);
      //});
    }

  } // Schedule::Schedule


  void
  Schedule::limitOutput(ParameterSet const& proc_pset,
                        BranchIDLists const& branchIDLists,
                        SubProcessParentageHelper const* subProcessParentageHelper) {
    std::string const output("output");

    ParameterSet const& maxEventsPSet = proc_pset.getUntrackedParameterSet("maxEvents");
    int maxEventSpecs = 0;
    int maxEventsOut = -1;
    ParameterSet const* vMaxEventsOut = nullptr;
    std::vector<std::string> intNamesE = maxEventsPSet.getParameterNamesForType<int>(false);
    if (search_all(intNamesE, output)) {
      maxEventsOut = maxEventsPSet.getUntrackedParameter<int>(output);
      ++maxEventSpecs;
    }
    std::vector<std::string> psetNamesE;
    maxEventsPSet.getParameterSetNames(psetNamesE, false);
    if (search_all(psetNamesE, output)) {
      vMaxEventsOut = &maxEventsPSet.getUntrackedParameterSet(output);
      ++maxEventSpecs;
    }

    if (maxEventSpecs > 1) {
      throw Exception(errors::Configuration) <<
        "\nAt most, one form of 'output' may appear in the 'maxEvents' parameter set";
    }

    for (auto& c : all_output_communicators_) {
      OutputModuleDescription desc(branchIDLists, maxEventsOut, subProcessParentageHelper);
      if (vMaxEventsOut != nullptr && !vMaxEventsOut->empty()) {
        std::string const& moduleLabel = c->description().moduleLabel();
        try {
          desc.maxEvents_ = vMaxEventsOut->getUntrackedParameter<int>(moduleLabel);
        } catch (Exception const&) {
          throw Exception(errors::Configuration) <<
            "\nNo entry in 'maxEvents' for output module label '" << moduleLabel << "'.\n";
        }
      }
      c->configure(desc);
    }
  }

  bool Schedule::terminate() const {
    if (all_output_communicators_.empty()) {
      return false;
    }
    for (auto& c : all_output_communicators_) {
      if (!c->limitReached()) {
        // Found an output module that has not reached output event count.
        return false;
      }
    }
    LogInfo("SuccessfulTermination")
      << "The job is terminating successfully because each output module\n"
      << "has reached its configured limit.\n";
    return true;
  }

  void Schedule::endJob(ExceptionCollector & collector) {
    globalSchedule_->endJob(collector);
    if (collector.hasThrown()) {
      return;
    }

    if (wantSummary_ == false) return;
    {
      TriggerReport tr;
      getTriggerReport(tr);

      // The trigger report (pass/fail etc.):

      LogVerbatim("FwkSummary") << "";
      if(streamSchedules_[0]->context().processContext()->isSubProcess()) {
        LogVerbatim("FwkSummary") << "TrigReport Process: "<<streamSchedules_[0]->context().processContext()->processName();
      }
      LogVerbatim("FwkSummary") << "TrigReport " << "---------- Event  Summary ------------";
      if(!tr.trigPathSummaries.empty()) {
        LogVerbatim("FwkSummary") << "TrigReport"
        << " Events total = " << tr.eventSummary.totalEvents
        << " passed = " << tr.eventSummary.totalEventsPassed
        << " failed = " << tr.eventSummary.totalEventsFailed
        << "";
      } else {
        LogVerbatim("FwkSummary") << "TrigReport"
        << " Events total = " << tr.eventSummary.totalEvents
        << " passed = " << tr.eventSummary.totalEvents
        << " failed = 0";
      }

      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "---------- Path   Summary ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
      << std::right << std::setw(10) << "Trig Bit#" << " "
      << std::right << std::setw(10) << "Executed" << " "
      << std::right << std::setw(10) << "Passed" << " "
      << std::right << std::setw(10) << "Failed" << " "
      << std::right << std::setw(10) << "Error" << " "
      << "Name" << "";
      for (auto const& p: tr.trigPathSummaries) {
        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(5) << 1
        << std::right << std::setw(5) << p.bitPosition << " "
        << std::right << std::setw(10) << p.timesRun << " "
        << std::right << std::setw(10) << p.timesPassed << " "
        << std::right << std::setw(10) << p.timesFailed << " "
        << std::right << std::setw(10) << p.timesExcept << " "
        << p.name << "";
      }

      /*
      std::vector<int>::const_iterator epi = empty_trig_paths_.begin();
      std::vector<int>::const_iterator epe = empty_trig_paths_.end();
      std::vector<std::string>::const_iterator  epn = empty_trig_path_names_.begin();
      for (; epi != epe; ++epi, ++epn) {

        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(5) << 1
        << std::right << std::setw(5) << *epi << " "
        << std::right << std::setw(10) << totalEvents() << " "
        << std::right << std::setw(10) << totalEvents() << " "
        << std::right << std::setw(10) << 0 << " "
        << std::right << std::setw(10) << 0 << " "
        << *epn << "";
      }
       */

      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "-------End-Path   Summary ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
      << std::right << std::setw(10) << "Trig Bit#" << " "
      << std::right << std::setw(10) << "Executed" << " "
      << std::right << std::setw(10) << "Passed" << " "
      << std::right << std::setw(10) << "Failed" << " "
      << std::right << std::setw(10) << "Error" << " "
      << "Name" << "";
      for (auto const& p: tr.endPathSummaries) {
        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(5) << 0
        << std::right << std::setw(5) << p.bitPosition << " "
        << std::right << std::setw(10) << p.timesRun << " "
        << std::right << std::setw(10) << p.timesPassed << " "
        << std::right << std::setw(10) << p.timesFailed << " "
        << std::right << std::setw(10) << p.timesExcept << " "
        << p.name << "";
      }

      for (auto const& p: tr.trigPathSummaries) {
        LogVerbatim("FwkSummary") << "";
        LogVerbatim("FwkSummary") << "TrigReport " << "---------- Modules in Path: " << p.name << " ------------";
        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(10) << "Trig Bit#" << " "
        << std::right << std::setw(10) << "Visited" << " "
        << std::right << std::setw(10) << "Passed" << " "
        << std::right << std::setw(10) << "Failed" << " "
        << std::right << std::setw(10) << "Error" << " "
        << "Name" << "";

        unsigned int bitpos = 0;
        for (auto const& mod: p.moduleInPathSummaries) {
          LogVerbatim("FwkSummary") << "TrigReport "
          << std::right << std::setw(5) << 1
          << std::right << std::setw(5) << bitpos << " "
          << std::right << std::setw(10) << mod.timesVisited << " "
          << std::right << std::setw(10) << mod.timesPassed << " "
          << std::right << std::setw(10) << mod.timesFailed << " "
          << std::right << std::setw(10) << mod.timesExcept << " "
          << mod.moduleLabel << "";
          ++bitpos;
        }
      }

      for (auto const& p: tr.endPathSummaries) {
        LogVerbatim("FwkSummary") << "";
        LogVerbatim("FwkSummary") << "TrigReport " << "------ Modules in End-Path: " << p.name << " ------------";
        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(10) << "Trig Bit#" << " "
        << std::right << std::setw(10) << "Visited" << " "
        << std::right << std::setw(10) << "Passed" << " "
        << std::right << std::setw(10) << "Failed" << " "
        << std::right << std::setw(10) << "Error" << " "
        << "Name" << "";

        unsigned int bitpos=0;
        for (auto const& mod: p.moduleInPathSummaries) {
          LogVerbatim("FwkSummary") << "TrigReport "
          << std::right << std::setw(5) << 0
          << std::right << std::setw(5) << bitpos << " "
          << std::right << std::setw(10) << mod.timesVisited << " "
          << std::right << std::setw(10) << mod.timesPassed << " "
          << std::right << std::setw(10) << mod.timesFailed << " "
          << std::right << std::setw(10) << mod.timesExcept << " "
          << mod.moduleLabel << "";
          ++bitpos;
        }
      }

      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "---------- Module Summary ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
      << std::right << std::setw(10) << "Visited" << " "
      << std::right << std::setw(10) << "Executed" << " "
      << std::right << std::setw(10) << "Passed" << " "
      << std::right << std::setw(10) << "Failed" << " "
      << std::right << std::setw(10) << "Error" << " "
      << "Name" << "";
      for (auto const& worker : tr.workerSummaries) {
        LogVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(10) << worker.timesVisited << " "
        << std::right << std::setw(10) << worker.timesRun << " "
        << std::right << std::setw(10) << worker.timesPassed << " "
        << std::right << std::setw(10) << worker.timesFailed << " "
        << std::right << std::setw(10) << worker.timesExcept << " "
        << worker.moduleLabel << "";
      }
      LogVerbatim("FwkSummary") << "";
    }
    // The timing report (CPU and Real Time):
    TriggerTimingReport tr;
    getTriggerTimingReport(tr);

    const int totalEvents = std::max(1, tr.eventSummary.totalEvents);

    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Event  Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport"
                              << std::setprecision(6) << std::fixed
                              << "       event loop CPU/event = " << tr.eventSummary.cpuTime/totalEvents;
    LogVerbatim("FwkSummary") << "TimeReport"
                              << std::setprecision(6) << std::fixed
                              << "      event loop Real/event = " << tr.eventSummary.realTime/totalEvents;
    LogVerbatim("FwkSummary") << "TimeReport"
                              << std::setprecision(6) << std::fixed
                              << "     sum Streams Real/event = " << tr.eventSummary.sumStreamRealTime/totalEvents;
    LogVerbatim("FwkSummary") << "TimeReport"
                              << std::setprecision(6) << std::fixed
                              << " efficiency CPU/Real/thread = " << tr.eventSummary.cpuTime/tr.eventSummary.realTime/preallocConfig_.numberOfThreads();

    constexpr int kColumn1Size = 10;
    constexpr int kColumn2Size = 12;
    constexpr int kColumn3Size = 12;
    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Path   Summary ---[Real sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event"<<" "
                              << std::right << std::setw(kColumn2Size) << "per exec"
                              << "  Name";
    for (auto const& p: tr.trigPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(kColumn1Size) << p.realTime/totalEvents << " "
                                << std::right << std::setw(kColumn2Size) << p.realTime/timesRun << "  "
                                << p.name << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event"<<" "
                              << std::right << std::setw(kColumn2Size) << "per exec"
                              << "  Name" << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "-------End-Path   Summary ---[Real sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event" <<" "
                              << std::right << std::setw(kColumn2Size) << "per exec"
                              << "  Name" << "";
    for (auto const& p: tr.endPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);

      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(kColumn1Size) << p.realTime/totalEvents << " "
                                << std::right << std::setw(kColumn2Size) << p.realTime/timesRun << "  "
                                << p.name << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event" <<" "
                              << std::right << std::setw(kColumn2Size) << "per exec"
                              << "  Name" << "";

    for (auto const& p: tr.trigPathSummaries) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "---------- Modules in Path: " << p.name << " ---[Real sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(kColumn1Size) << "per event" <<" "
                                << std::right << std::setw(kColumn2Size) << "per visit"
                                << "  Name" << "";
      for (auto const& mod: p.moduleInPathSummaries) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(kColumn1Size) << mod.realTime/totalEvents << " "
                                  << std::right << std::setw(kColumn2Size) << mod.realTime/std::max(1, mod.timesVisited) << "  "
                                  << mod.moduleLabel << "";
      }
    }
    if(not tr.trigPathSummaries.empty()) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(kColumn1Size) << "per event" <<" "
                                << std::right << std::setw(kColumn2Size) << "per visit"
                                << "  Name" << "";
    }
    for (auto const& p: tr.endPathSummaries) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "------ Modules in End-Path: " << p.name << " ---[Real sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(kColumn1Size) << "per event" <<" "
                                << std::right << std::setw(kColumn2Size) << "per visit"
                                << "  Name" << "";
      for (auto const& mod: p.moduleInPathSummaries) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(kColumn1Size) << mod.realTime/totalEvents << " "
                                  << std::right << std::setw(kColumn2Size) << mod.realTime/std::max(1, mod.timesVisited) << "  "
                                  << mod.moduleLabel << "";
      }
    }
    if(not tr.endPathSummaries.empty()) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(kColumn1Size) << "per event" <<" "
                                << std::right << std::setw(kColumn2Size) << "per visit"
                                << "  Name" << "";
    }
    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Module Summary ---[Real sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event" <<" "
                              << std::right << std::setw(kColumn2Size) << "per exec" <<" "
                              << std::right << std::setw(kColumn3Size) << "per visit"
                              << "  Name" << "";
    for (auto const& worker : tr.workerSummaries) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(kColumn1Size) << worker.realTime/totalEvents << " "
                                << std::right << std::setw(kColumn2Size) << worker.realTime/std::max(1, worker.timesRun) << " "
                                << std::right << std::setw(kColumn3Size) << worker.realTime/std::max(1, worker.timesVisited) << "  "
                                << worker.moduleLabel << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(kColumn1Size) << "per event" <<" "
                              << std::right << std::setw(kColumn2Size) << "per exec" <<" "
                              << std::right << std::setw(kColumn3Size) << "per visit"
                              << "  Name" << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "T---Report end!" << "";
    LogVerbatim("FwkSummary") << "";
  }

  void Schedule::closeOutputFiles() {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::closeFile, _1));
  }

  void Schedule::openOutputFiles(FileBlock& fb) {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::openFile, _1, std::cref(fb)));
  }

  void Schedule::writeRunAsync(WaitingTaskHolder task,
                               RunPrincipal const& rp,
                               ProcessContext const* processContext,
                               ActivityRegistry* activityRegistry,
                               MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    for(auto& c: all_output_communicators_) {
      c->writeRunAsync(task, rp, processContext, activityRegistry, mergeableRunProductMetadata);
    }
  }

  void Schedule::writeLumiAsync(WaitingTaskHolder task,
                                LuminosityBlockPrincipal const& lbp,
                                ProcessContext const* processContext,
                                ActivityRegistry* activityRegistry) {
    for(auto& c: all_output_communicators_) {
      c->writeLumiAsync(task, lbp, processContext, activityRegistry);
    }
  }

  bool Schedule::shouldWeCloseOutput() const {
    using std::placeholders::_1;
    // Return true iff at least one output module returns true.
    return (std::find_if (all_output_communicators_.begin(), all_output_communicators_.end(),
                     std::bind(&OutputModuleCommunicator::shouldWeCloseFile, _1))
                     != all_output_communicators_.end());
  }

  void Schedule::respondToOpenInputFile(FileBlock const& fb) {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::respondToOpenInputFile, _1, std::cref(fb)));
  }

  void Schedule::respondToCloseInputFile(FileBlock const& fb) {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::respondToCloseInputFile, _1, std::cref(fb)));
  }

  void Schedule::beginJob(ProductRegistry const& iRegistry) {
    globalSchedule_->beginJob(iRegistry);
  }

  void Schedule::beginStream(unsigned int iStreamID) {
    assert(iStreamID<streamSchedules_.size());
    streamSchedules_[iStreamID]->beginStream();
  }

  void Schedule::endStream(unsigned int iStreamID) {
    assert(iStreamID<streamSchedules_.size());
    streamSchedules_[iStreamID]->endStream();
  }
  
  void Schedule::processOneEventAsync(WaitingTaskHolder iTask,
                                      unsigned int iStreamID,
                                      EventPrincipal& ep,
                                      EventSetup const& es,
                                      ServiceToken const& token) {
    assert(iStreamID<streamSchedules_.size());
    streamSchedules_[iStreamID]->processOneEventAsync(std::move(iTask),ep,es,token,pathStatusInserters_);
  }
  
  bool Schedule::changeModule(std::string const& iLabel,
                              ParameterSet const& iPSet,
                              const ProductRegistry& iRegistry) {
    Worker* found = nullptr;
    for (auto const& worker : allWorkers()) {
      if (worker->description().moduleLabel() == iLabel) {
        found = worker;
        break;
      }
    }
    if (nullptr == found) {
      return false;
    }

    auto newMod = moduleRegistry_->replaceModule(iLabel,iPSet,preallocConfig_);

    globalSchedule_->replaceModule(newMod,iLabel);

    for(auto& s: streamSchedules_) {
      s->replaceModule(newMod,iLabel);
    }

    {
      //Need to updateLookup in order to make getByToken work
      auto const runLookup = iRegistry.productLookup(InRun);
      auto const lumiLookup = iRegistry.productLookup(InLumi);
      auto const eventLookup = iRegistry.productLookup(InEvent);
      found->updateLookup(InRun,*runLookup);
      found->updateLookup(InLumi,*lumiLookup);
      found->updateLookup(InEvent,*eventLookup);
      
      auto const& processName = newMod->moduleDescription().processName();
      auto const& runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
      auto const& lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
      auto const& eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);
      found->resolvePutIndicies(InRun,runModuleToIndicies);
      found->resolvePutIndicies(InLumi,lumiModuleToIndicies);
      found->resolvePutIndicies(InEvent,eventModuleToIndicies);


    }

    return true;
  }

  std::vector<ModuleDescription const*>
  Schedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->descPtr();
      result.push_back(p);
    }
    return result;
  }

  Schedule::AllWorkers const&
  Schedule::allWorkers() const {
    return globalSchedule_->allWorkers();
  }

  void Schedule::convertCurrentProcessAlias(std::string const& processName) {
    for (auto const& worker : allWorkers()) {
      worker->convertCurrentProcessAlias(processName);
    }
  }

  void
  Schedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->availablePaths(oLabelsToFill);
  }

  void
  Schedule::triggerPaths(std::vector<std::string>& oLabelsToFill) const {
    oLabelsToFill = *pathNames_;

  }

  void
  Schedule::endPaths(std::vector<std::string>& oLabelsToFill) const {
    oLabelsToFill = *endPathNames_;
  }

  void
  Schedule::modulesInPath(std::string const& iPathLabel,
                          std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->modulesInPath(iPathLabel,oLabelsToFill);
  }

  void
  Schedule::moduleDescriptionsInPath(std::string const& iPathLabel,
                                     std::vector<ModuleDescription const*>& descriptions,
                                     unsigned int hint) const {
    streamSchedules_[0]->moduleDescriptionsInPath(iPathLabel, descriptions, hint);
  }

  void
  Schedule::moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                        std::vector<ModuleDescription const*>& descriptions,
                                        unsigned int hint) const {
    streamSchedules_[0]->moduleDescriptionsInEndPath(iEndPathLabel, descriptions, hint);
  }

  void
  Schedule::fillModuleAndConsumesInfo(std::vector<ModuleDescription const*>& allModuleDescriptions,
                                      std::vector<std::pair<unsigned int, unsigned int> >& moduleIDToIndex,
                                      std::vector<std::vector<ModuleDescription const*> >& modulesWhoseProductsAreConsumedBy,
                                      ProductRegistry const& preg) const {
    allModuleDescriptions.clear();
    moduleIDToIndex.clear();
    modulesWhoseProductsAreConsumedBy.clear();

    allModuleDescriptions.reserve(allWorkers().size());
    moduleIDToIndex.reserve(allWorkers().size());
    modulesWhoseProductsAreConsumedBy.resize(allWorkers().size());

    std::map<std::string, ModuleDescription const*> labelToDesc;
    unsigned int i = 0;
    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->descPtr();
      allModuleDescriptions.push_back(p);
      moduleIDToIndex.push_back(std::pair<unsigned int, unsigned int>(p->id(), i));
      labelToDesc[p->moduleLabel()] = p;
      ++i;
    }
    sort_all(moduleIDToIndex);

    i = 0;
    for (auto const& worker : allWorkers()) {
      std::vector<ModuleDescription const*>& modules = modulesWhoseProductsAreConsumedBy.at(i);
      worker->modulesWhoseProductsAreConsumed(modules, preg, labelToDesc);
      ++i;
    }
  }

  void
  Schedule::enableEndPaths(bool active) {
    endpathsAreActive_ = active;
    for(auto& s : streamSchedules_) {
      s->enableEndPaths(active);
    }
  }

  bool
  Schedule::endPathsEnabled() const {
    return endpathsAreActive_;
  }

  void
  Schedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents = 0;
    rep.eventSummary.totalEventsPassed = 0;
    rep.eventSummary.totalEventsFailed = 0;
    for(auto& s: streamSchedules_) {
      s->getTriggerReport(rep);
    }
    sort_all(rep.workerSummaries);
  }

  void
  Schedule::getTriggerTimingReport(TriggerTimingReport& rep) const {
    rep.eventSummary.totalEvents = 0;
    rep.eventSummary.cpuTime = 0.;
    rep.eventSummary.realTime = 0.;
    summaryTimeKeeper_->fillTriggerTimingReport(rep);
  }

  int
  Schedule::totalEvents() const {
    int returnValue = 0;
    for(auto& s: streamSchedules_) {
      returnValue += s->totalEvents();
    }
    return returnValue;
  }

  int
  Schedule::totalEventsPassed() const {
    int returnValue = 0;
    for(auto& s: streamSchedules_) {
      returnValue += s->totalEventsPassed();
    }
    return returnValue;
  }

  int
  Schedule::totalEventsFailed() const {
    int returnValue = 0;
    for(auto& s: streamSchedules_) {
      returnValue += s->totalEventsFailed();
    }
    return returnValue;
  }


  void
  Schedule::clearCounters() {
    for(auto& s: streamSchedules_) {
      s->clearCounters();
    }
  }
}
