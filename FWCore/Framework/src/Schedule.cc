#include "FWCore/Framework/interface/Schedule.h"

#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/TriggerReport.h"
#include "FWCore/Framework/src/TriggerTimingReport.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/OutputModuleCommunicator.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/chain_first.h"
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
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <array>
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
#include "processEDAliases.h"

namespace edm {

  class Maker;

  namespace {
    using std::placeholders::_1;

    bool binary_search_string(std::vector<std::string> const& v, std::string const& s) {
      return std::binary_search(v.begin(), v.end(), s);
    }

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    std::shared_ptr<TriggerResultInserter> makeInserter(
        ParameterSet& proc_pset,
        PreallocationConfiguration const& iPrealloc,
        ProductRegistry& preg,
        ExceptionToActionTable const& actions,
        std::shared_ptr<ActivityRegistry> areg,
        std::shared_ptr<ProcessConfiguration const> processConfiguration) {
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
      // Caught exception is rethrown
      CMS_SA_ALLOW try {
        maker::ModuleHolderT<TriggerResultInserter> holder(
            make_shared_noexcept_false<TriggerResultInserter>(*trig_pset, iPrealloc.numberOfStreams()),
            static_cast<Maker const*>(nullptr));
        holder.setModuleDescription(md);
        holder.registerProductsAndCallbacks(&preg);
        returnValue = holder.module();
        postCalled = true;
        // if exception then post will be called in the catch block
        areg->postModuleConstructionSignal_(md);
      } catch (...) {
        if (!postCalled) {
          CMS_SA_ALLOW try { areg->postModuleConstructionSignal_(md); } catch (...) {
            // If post throws an exception ignore it because we are already handling another exception
          }
        }
        throw;
      }
      return returnValue;
    }

    template <typename T>
    void makePathStatusInserters(std::vector<edm::propagate_const<std::shared_ptr<T>>>& pathStatusInserters,
                                 std::vector<std::string> const& pathNames,
                                 PreallocationConfiguration const& iPrealloc,
                                 ProductRegistry& preg,
                                 std::shared_ptr<ActivityRegistry> areg,
                                 std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                 std::string const& moduleTypeName) {
      ParameterSet pset;
      pset.addParameter<std::string>("@module_type", moduleTypeName);
      pset.addParameter<std::string>("@module_edm_type", "EDProducer");
      pset.registerIt();

      pathStatusInserters.reserve(pathNames.size());

      for (auto const& pathName : pathNames) {
        ModuleDescription md(
            pset.id(), moduleTypeName, pathName, processConfiguration.get(), ModuleDescription::getUniqueID());

        areg->preModuleConstructionSignal_(md);
        bool postCalled = false;
        // Caught exception is rethrown
        CMS_SA_ALLOW try {
          maker::ModuleHolderT<T> holder(make_shared_noexcept_false<T>(iPrealloc.numberOfStreams()),
                                         static_cast<Maker const*>(nullptr));
          holder.setModuleDescription(md);
          holder.registerProductsAndCallbacks(&preg);
          pathStatusInserters.emplace_back(holder.module());
          postCalled = true;
          // if exception then post will be called in the catch block
          areg->postModuleConstructionSignal_(md);
        } catch (...) {
          if (!postCalled) {
            CMS_SA_ALLOW try { areg->postModuleConstructionSignal_(md); } catch (...) {
              // If post throws an exception ignore it because we are already handling another exception
            }
          }
          throw;
        }
      }
    }

    typedef std::vector<std::string> vstring;

    void processSwitchProducers(ParameterSet const& proc_pset, std::string const& processName, ProductRegistry& preg) {
      // Update Switch BranchDescriptions for the chosen case
      struct BranchesCases {
        BranchesCases(std::vector<std::string> cases) : caseLabels{std::move(cases)} {}
        std::vector<BranchKey> chosenBranches;
        std::vector<std::string> caseLabels;
      };
      std::map<std::string, BranchesCases> switchMap;
      for (auto& prod : preg.productListUpdator()) {
        if (prod.second.isSwitchAlias()) {
          auto it = switchMap.find(prod.second.moduleLabel());
          if (it == switchMap.end()) {
            auto const& switchPSet = proc_pset.getParameter<edm::ParameterSet>(prod.second.moduleLabel());
            auto inserted = switchMap.emplace(prod.second.moduleLabel(),
                                              switchPSet.getParameter<std::vector<std::string>>("@all_cases"));
            assert(inserted.second);
            it = inserted.first;
          }

          bool found = false;
          for (auto const& productIter : preg.productList()) {
            BranchKey const& branchKey = productIter.first;
            // The alias-for product must be in the same process as
            // the SwitchProducer (earlier processes or SubProcesses
            // may contain products with same type, module label, and
            // instance name)
            if (branchKey.processName() != processName) {
              continue;
            }

            BranchDescription const& desc = productIter.second;
            if (desc.branchType() == prod.second.branchType() and
                desc.unwrappedTypeID().typeInfo() == prod.second.unwrappedTypeID().typeInfo() and
                branchKey.moduleLabel() == prod.second.switchAliasModuleLabel() and
                branchKey.productInstanceName() == prod.second.productInstanceName()) {
              prod.second.setSwitchAliasForBranch(desc);
              if (!prod.second.transient()) {
                it->second.chosenBranches.push_back(prod.first);  // with moduleLabel of the Switch
              }
              found = true;
            }
          }
          if (not found) {
            Exception ex(errors::LogicError);
            ex << "Trying to find a BranchDescription to be aliased-for by SwitchProducer with\n"
               << "  friendly class name = " << prod.second.friendlyClassName() << "\n"
               << "  module label = " << prod.second.moduleLabel() << "\n"
               << "  product instance name = " << prod.second.productInstanceName() << "\n"
               << "  process name = " << processName
               << "\n\nbut did not find any. Please contact a framework developer.";
            ex.addContext("Calling Schedule.cc:processSwitchProducers()");
            throw ex;
          }
        }
      }
      if (switchMap.empty())
        return;

      for (auto& elem : switchMap) {
        std::sort(elem.second.chosenBranches.begin(), elem.second.chosenBranches.end());
      }

      auto addProductsToException = [&preg, &processName](auto const& caseLabels, edm::Exception& ex) {
        std::map<std::string, std::vector<BranchKey>> caseBranches;
        for (auto const& item : preg.productList()) {
          if (item.first.processName() != processName)
            continue;

          if (auto found = std::find(caseLabels.begin(), caseLabels.end(), item.first.moduleLabel());
              found != caseLabels.end()) {
            caseBranches[*found].push_back(item.first);
          }
        }

        for (auto const& caseLabel : caseLabels) {
          ex << "Products for case " << caseLabel << " (friendly class name, product instance name):\n";
          auto& branches = caseBranches[caseLabel];
          std::sort(branches.begin(), branches.end());
          for (auto const& branch : branches) {
            ex << " " << branch.friendlyClassName() << " " << branch.productInstanceName() << "\n";
          }
        }
      };

      // Check that non-chosen cases declare exactly the same non-transient branches
      // Also set the alias-for branches to transient
      std::vector<bool> foundBranches;
      for (auto const& switchItem : switchMap) {
        auto const& switchLabel = switchItem.first;
        auto const& chosenBranches = switchItem.second.chosenBranches;
        auto const& caseLabels = switchItem.second.caseLabels;
        foundBranches.resize(chosenBranches.size());
        for (auto const& caseLabel : caseLabels) {
          std::fill(foundBranches.begin(), foundBranches.end(), false);
          for (auto& nonConstItem : preg.productListUpdator()) {
            auto const& item = nonConstItem;
            if (item.first.moduleLabel() == caseLabel and item.first.processName() == processName) {
              // Check that products which are not transient in the dictionary are consistent between
              // all the cases of a SwitchProducer.
              if (!item.second.transient()) {
                auto range = std::equal_range(chosenBranches.begin(),
                                              chosenBranches.end(),
                                              BranchKey(item.first.friendlyClassName(),
                                                        switchLabel,
                                                        item.first.productInstanceName(),
                                                        item.first.processName()));
                if (range.first == range.second) {
                  Exception ex(errors::Configuration);
                  ex << "SwitchProducer " << switchLabel << " has a case " << caseLabel << " with a product "
                     << item.first << " that is not produced by the chosen case "
                     << proc_pset.getParameter<edm::ParameterSet>(switchLabel)
                            .getUntrackedParameter<std::string>("@chosen_case")
                     << " and that product is not transient. "
                     << "If the intention is to produce only a subset of the non-transient products listed below, each "
                        "case with more non-transient products needs to be replaced with an EDAlias to only the "
                        "necessary products, and the EDProducer itself needs to be moved to a Task.\n\n";
                  addProductsToException(caseLabels, ex);
                  throw ex;
                }
                assert(std::distance(range.first, range.second) == 1);
                foundBranches[std::distance(chosenBranches.begin(), range.first)] = true;
              }

              // Set the alias-for branch as transient so it gets fully ignored in output.
              // I tried first to implicitly drop all branches with
              // '@' in ProductSelector, but that gave problems on
              // input (those branches would be implicitly dropped on
              // input as well, leading to the SwitchProducer branches
              // do be dropped as dependent ones, as the alias
              // detection logic in RootFile says that the
              // SwitchProducer branches are not alias branches)
              nonConstItem.second.setTransient(true);

              // Check that there are no BranchAliases for any of the cases
              auto const& bd = item.second;
              if (not bd.branchAliases().empty()) {
                auto ex = Exception(errors::UnimplementedFeature)
                          << "SwitchProducer does not support ROOT branch aliases. Got the following ROOT branch "
                             "aliases for SwitchProducer with label "
                          << switchLabel << " for case " << caseLabel << ":";
                for (auto const& branchAlias : bd.branchAliases()) {
                  ex << " " << branchAlias;
                }
                throw ex;
              }
            }
          }

          for (size_t i = 0; i < chosenBranches.size(); i++) {
            if (not foundBranches[i]) {
              auto chosenLabel = proc_pset.getParameter<edm::ParameterSet>(switchLabel)
                                     .getUntrackedParameter<std::string>("@chosen_case");
              Exception ex(errors::Configuration);
              ex << "SwitchProducer " << switchLabel << " has a case " << caseLabel
                 << " that does not produce a product " << chosenBranches[i] << " that is produced by the chosen case "
                 << chosenLabel << " and that product is not transient. "
                 << "If the intention is to produce only a subset of the non-transient products listed below, each "
                    "case with more non-transient products needs to be replaced with an EDAlias to only the "
                    "necessary products, and the EDProducer itself needs to be moved to a Task.\n\n";
              addProductsToException(caseLabels, ex);
              throw ex;
            }
          }
        }
      }
    }

    void reduceParameterSet(ParameterSet& proc_pset,
                            vstring const& end_path_name_list,
                            vstring& modulesInConfig,
                            std::set<std::string> const& usedModuleLabels,
                            std::map<std::string, std::vector<std::pair<std::string, int>>>& outputModulePathPositions) {
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
        std::set<std::string> noEndPaths(scheduledPaths.begin(), scheduledPaths.end());
        for (auto const& endPath : end_path_name_list) {
          noEndPaths.erase(endPath);
        }
        {
          vstring labels;
          for (auto const& path : noEndPaths) {
            labels = proc_pset.getParameter<vstring>(path);
            modulesOnPaths.insert(labels.begin(), labels.end());
          }
        }
      }
      //Initially fill labelsToBeDropped with all module mentioned in
      // the configuration but which are not being used by the system
      std::vector<std::string> labelsToBeDropped;
      labelsToBeDropped.reserve(modulesInConfigSet.size());
      std::set_difference(modulesInConfigSet.begin(),
                          modulesInConfigSet.end(),
                          usedModuleLabels.begin(),
                          usedModuleLabels.end(),
                          std::back_inserter(labelsToBeDropped));

      const unsigned int sizeBeforeOutputModules = labelsToBeDropped.size();
      for (auto const& modLabel : usedModuleLabels) {
        // Do nothing for modules that do not have a ParameterSet. Modules of type
        // PathStatusInserter and EndPathStatusInserter will not have a ParameterSet.
        if (proc_pset.existsAs<ParameterSet>(modLabel)) {
          edmType = proc_pset.getParameterSet(modLabel).getParameter<std::string>(moduleEdmType);
          if (edmType == outputModule) {
            outputModuleLabels.push_back(modLabel);
            labelsToBeDropped.push_back(modLabel);
          }
          if (edmType == edAnalyzer) {
            if (modulesOnPaths.end() == modulesOnPaths.find(modLabel)) {
              labelsToBeDropped.push_back(modLabel);
            }
          }
        }
      }
      //labelsToBeDropped must be sorted
      std::inplace_merge(
          labelsToBeDropped.begin(), labelsToBeDropped.begin() + sizeBeforeOutputModules, labelsToBeDropped.end());

      // drop the parameter sets used to configure the modules
      for_all(labelsToBeDropped, std::bind(&ParameterSet::eraseOrSetUntrackedParameterSet, std::ref(proc_pset), _1));

      // drop the labels from @all_modules
      vstring::iterator endAfterRemove =
          std::remove_if(modulesInConfig.begin(),
                         modulesInConfig.end(),
                         std::bind(binary_search_string, std::ref(labelsToBeDropped), _1));
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

        for (vstring::iterator iLabel = labels.begin(), iEnd = labels.end(); iLabel != iEnd; ++iLabel) {
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
      endAfterRemove = std::remove_if(scheduledPaths.begin(),
                                      scheduledPaths.end(),
                                      std::bind(binary_search_string, std::ref(endPathsToBeDropped), _1));
      scheduledPaths.erase(endAfterRemove, scheduledPaths.end());
      proc_pset.addParameter<vstring>(std::string("@paths"), scheduledPaths);

      // remove empty end paths from @end_paths
      vstring scheduledEndPaths = proc_pset.getParameter<vstring>("@end_paths");
      endAfterRemove = std::remove_if(scheduledEndPaths.begin(),
                                      scheduledEndPaths.end(),
                                      std::bind(binary_search_string, std::ref(endPathsToBeDropped), _1));
      scheduledEndPaths.erase(endAfterRemove, scheduledEndPaths.end());
      proc_pset.addParameter<vstring>(std::string("@end_paths"), scheduledEndPaths);
    }

    class RngEDConsumer : public EDConsumerBase {
    public:
      explicit RngEDConsumer(std::set<TypeID>& typesConsumed) {
        Service<RandomNumberGenerator> rng;
        if (rng.isAvailable()) {
          rng->consumes(consumesCollector());
          for (auto const& consumesInfo : this->consumesInfo()) {
            typesConsumed.emplace(consumesInfo.type());
          }
        }
      }
    };

    template <typename F>
    auto doCleanup(F&& iF) {
      auto wrapped = [f = std::move(iF)](std::exception_ptr const* iPtr, edm::WaitingTaskHolder iTask) {
        CMS_SA_ALLOW try { f(); } catch (...) {
        }
        if (iPtr) {
          iTask.doneWaiting(*iPtr);
        }
      };
      return wrapped;
    }
  }  // namespace
  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  Schedule::Schedule(ParameterSet& proc_pset,
                     service::TriggerNamesService const& tns,
                     ProductRegistry& preg,
                     ExceptionToActionTable const& actions,
                     std::shared_ptr<ActivityRegistry> areg,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration,
                     PreallocationConfiguration const& prealloc,
                     ProcessContext const* processContext,
                     ModuleTypeResolverMaker const* resolverMaker)
      :  //Only create a resultsInserter if there is a trigger path
        resultsInserter_{tns.getTrigPaths().empty()
                             ? std::shared_ptr<TriggerResultInserter>{}
                             : makeInserter(proc_pset, prealloc, preg, actions, areg, processConfiguration)},
        moduleRegistry_(std::make_shared<ModuleRegistry>(resolverMaker)),
        all_output_communicators_(),
        preallocConfig_(prealloc),
        pathNames_(&tns.getTrigPaths()),
        endPathNames_(&tns.getEndPaths()),
        wantSummary_(tns.wantSummary()) {
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

    assert(0 < prealloc.numberOfStreams());
    streamSchedules_.reserve(prealloc.numberOfStreams());
    for (unsigned int i = 0; i < prealloc.numberOfStreams(); ++i) {
      streamSchedules_.emplace_back(make_shared_noexcept_false<StreamSchedule>(resultsInserter(),
                                                                               pathStatusInserters_,
                                                                               endPathStatusInserters_,
                                                                               moduleRegistry(),
                                                                               proc_pset,
                                                                               tns,
                                                                               prealloc,
                                                                               preg,
                                                                               actions,
                                                                               areg,
                                                                               processConfiguration,
                                                                               StreamID{i},
                                                                               processContext));
    }

    //TriggerResults are injected automatically by StreamSchedules and are
    // unknown to the ModuleRegistry
    const std::string kTriggerResults("TriggerResults");
    std::vector<std::string> modulesToUse;
    modulesToUse.reserve(streamSchedules_[0]->allWorkers().size());
    for (auto const& worker : streamSchedules_[0]->allWorkers()) {
      if (worker->description()->moduleLabel() != kTriggerResults) {
        modulesToUse.push_back(worker->description()->moduleLabel());
      }
    }
    //The unscheduled modules are at the end of the list, but we want them at the front
    unsigned int const nUnscheduledModules = streamSchedules_[0]->numberOfUnscheduledModules();
    if (nUnscheduledModules > 0) {
      std::vector<std::string> temp;
      temp.reserve(modulesToUse.size());
      auto itBeginUnscheduled = modulesToUse.begin() + modulesToUse.size() - nUnscheduledModules;
      std::copy(itBeginUnscheduled, modulesToUse.end(), std::back_inserter(temp));
      std::copy(modulesToUse.begin(), itBeginUnscheduled, std::back_inserter(temp));
      temp.swap(modulesToUse);
    }

    // propagate_const<T> has no reset() function
    globalSchedule_ = std::make_unique<GlobalSchedule>(resultsInserter(),
                                                       pathStatusInserters_,
                                                       endPathStatusInserters_,
                                                       moduleRegistry(),
                                                       modulesToUse,
                                                       proc_pset,
                                                       preg,
                                                       prealloc,
                                                       actions,
                                                       areg,
                                                       processConfiguration,
                                                       processContext);
  }

  void Schedule::finishSetup(ParameterSet& proc_pset,
                             service::TriggerNamesService const& tns,
                             ProductRegistry& preg,
                             BranchIDListHelper& branchIDListHelper,
                             ProcessBlockHelperBase& processBlockHelper,
                             ThinnedAssociationsHelper& thinnedAssociationsHelper,
                             SubProcessParentageHelper const* subProcessParentageHelper,
                             std::shared_ptr<ActivityRegistry> areg,
                             std::shared_ptr<ProcessConfiguration> processConfiguration,
                             bool hasSubprocesses,
                             PreallocationConfiguration const& prealloc,
                             ProcessContext const* processContext) {
    //TriggerResults is not in the top level ParameterSet so the call to
    // reduceParameterSet would fail to find it. Just remove it up front.
    const std::string kTriggerResults("TriggerResults");

    std::set<std::string> usedModuleLabels;
    for (auto const& worker : allWorkers()) {
      if (worker->description()->moduleLabel() != kTriggerResults) {
        usedModuleLabels.insert(worker->description()->moduleLabel());
      }
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string>>("@all_modules"));
    std::map<std::string, std::vector<std::pair<std::string, int>>> outputModulePathPositions;
    reduceParameterSet(proc_pset, tns.getEndPaths(), modulesInConfig, usedModuleLabels, outputModulePathPositions);
    {
      std::vector<std::string> aliases = proc_pset.getParameter<std::vector<std::string>>("@all_aliases");
      detail::processEDAliases(aliases, {}, proc_pset, processConfiguration->processName(), preg);
    }

    // At this point all BranchDescriptions are created. Mark now the
    // ones of unscheduled workers to be on-demand.
    {
      auto const& unsched = streamSchedules_[0]->unscheduledWorkers();
      if (not unsched.empty()) {
        std::set<std::string> unscheduledModules;
        std::transform(unsched.begin(),
                       unsched.end(),
                       std::insert_iterator<std::set<std::string>>(unscheduledModules, unscheduledModules.begin()),
                       [](auto worker) { return worker->description()->moduleLabel(); });
        preg.setUnscheduledProducts(unscheduledModules);
      }
    }

    processSwitchProducers(proc_pset, processConfiguration->processName(), preg);
    proc_pset.registerIt();
    processConfiguration->setParameterSetID(proc_pset.id());
    processConfiguration->setProcessConfigurationID();

    // This is used for a little sanity-check to make sure no code
    // modifications alter the number of workers at a later date.
    size_t all_workers_count = allWorkers().size();

    moduleRegistry_->forAllModuleHolders([this](maker::ModuleHolder* iHolder) {
      auto comm = iHolder->createOutputModuleCommunicator();
      if (comm) {
        all_output_communicators_.emplace_back(std::shared_ptr<OutputModuleCommunicator>{comm.release()});
      }
    });
    // Now that the output workers are filled in, set any output limits or information.
    limitOutput(proc_pset, branchIDListHelper.branchIDLists(), subProcessParentageHelper);

    // Sanity check: make sure nobody has added a worker after we've
    // already relied on the WorkerManager being full.
    assert(all_workers_count == allWorkers().size());

    branchIDListHelper.updateFromRegistry(preg);

    for (auto const& worker : streamSchedules_[0]->allWorkers()) {
      worker->registerThinnedAssociations(preg, thinnedAssociationsHelper);
    }

    processBlockHelper.updateForNewProcess(preg, processConfiguration->processName());

    // The output modules consume products in kept branches.
    // So we must set this up before freezing.
    for (auto& c : all_output_communicators_) {
      c->selectProducts(preg, thinnedAssociationsHelper, processBlockHelper);
    }

    for (auto& product : preg.productListUpdator()) {
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
      if (hasSubprocesses) {
        productTypesConsumed.emplace(typeid(TriggerResults));
      }
      // The RandomNumberGeneratorService is not a module, yet it consumes.
      { RngEDConsumer rngConsumer = RngEDConsumer(productTypesConsumed); }
      preg.setFrozen(productTypesConsumed, elementTypesConsumed, processConfiguration->processName());
    }

    for (auto& c : all_output_communicators_) {
      c->setEventSelectionInfo(outputModulePathPositions, preg.anyProductProduced());
    }

    if (wantSummary_) {
      std::vector<const ModuleDescription*> modDesc;
      const auto& workers = allWorkers();
      modDesc.reserve(workers.size());

      std::transform(workers.begin(),
                     workers.end(),
                     std::back_inserter(modDesc),
                     [](const Worker* iWorker) -> const ModuleDescription* { return iWorker->description(); });

      // propagate_const<T> has no reset() function
      summaryTimeKeeper_ = std::make_unique<SystemTimeKeeper>(prealloc.numberOfStreams(), modDesc, tns, processContext);
      auto timeKeeperPtr = summaryTimeKeeper_.get();

      areg->watchPreModuleDestruction(timeKeeperPtr, &SystemTimeKeeper::removeModuleIfExists);

      areg->watchPreModuleEvent(timeKeeperPtr, &SystemTimeKeeper::startModuleEvent);
      areg->watchPostModuleEvent(timeKeeperPtr, &SystemTimeKeeper::stopModuleEvent);
      areg->watchPreModuleEventAcquire(timeKeeperPtr, &SystemTimeKeeper::restartModuleEvent);
      areg->watchPostModuleEventAcquire(timeKeeperPtr, &SystemTimeKeeper::stopModuleEvent);
      areg->watchPreModuleEventDelayedGet(timeKeeperPtr, &SystemTimeKeeper::pauseModuleEvent);
      areg->watchPostModuleEventDelayedGet(timeKeeperPtr, &SystemTimeKeeper::restartModuleEvent);

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

  }  // Schedule::Schedule

  void Schedule::limitOutput(ParameterSet const& proc_pset,
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
      throw Exception(errors::Configuration)
          << "\nAt most, one form of 'output' may appear in the 'maxEvents' parameter set";
    }

    for (auto& c : all_output_communicators_) {
      OutputModuleDescription desc(branchIDLists, maxEventsOut, subProcessParentageHelper);
      if (vMaxEventsOut != nullptr && !vMaxEventsOut->empty()) {
        std::string const& moduleLabel = c->description().moduleLabel();
        try {
          desc.maxEvents_ = vMaxEventsOut->getUntrackedParameter<int>(moduleLabel);
        } catch (Exception const&) {
          throw Exception(errors::Configuration)
              << "\nNo entry in 'maxEvents' for output module label '" << moduleLabel << "'.\n";
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
    LogInfo("SuccessfulTermination") << "The job is terminating successfully because each output module\n"
                                     << "has reached its configured limit.\n";
    return true;
  }

  void Schedule::endJob(ExceptionCollector& collector) {
    globalSchedule_->endJob(collector);
    if (collector.hasThrown()) {
      return;
    }

    if (wantSummary_ == false)
      return;
    {
      TriggerReport tr;
      getTriggerReport(tr);

      // The trigger report (pass/fail etc.):

      LogFwkVerbatim("FwkSummary") << "";
      if (streamSchedules_[0]->context().processContext()->isSubProcess()) {
        LogFwkVerbatim("FwkSummary") << "TrigReport Process: "
                                     << streamSchedules_[0]->context().processContext()->processName();
      }
      LogFwkVerbatim("FwkSummary") << "TrigReport "
                                   << "---------- Event  Summary ------------";
      if (!tr.trigPathSummaries.empty()) {
        LogFwkVerbatim("FwkSummary") << "TrigReport"
                                     << " Events total = " << tr.eventSummary.totalEvents
                                     << " passed = " << tr.eventSummary.totalEventsPassed
                                     << " failed = " << tr.eventSummary.totalEventsFailed << "";
      } else {
        LogFwkVerbatim("FwkSummary") << "TrigReport"
                                     << " Events total = " << tr.eventSummary.totalEvents
                                     << " passed = " << tr.eventSummary.totalEvents << " failed = 0";
      }

      LogFwkVerbatim("FwkSummary") << "";
      LogFwkVerbatim("FwkSummary") << "TrigReport "
                                   << "---------- Path   Summary ------------";
      LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << "Trig Bit#"
                                   << " " << std::right << std::setw(10) << "Executed"
                                   << " " << std::right << std::setw(10) << "Passed"
                                   << " " << std::right << std::setw(10) << "Failed"
                                   << " " << std::right << std::setw(10) << "Error"
                                   << " "
                                   << "Name"
                                   << "";
      for (auto const& p : tr.trigPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(5) << 1 << std::right << std::setw(5)
                                     << p.bitPosition << " " << std::right << std::setw(10) << p.timesRun << " "
                                     << std::right << std::setw(10) << p.timesPassed << " " << std::right
                                     << std::setw(10) << p.timesFailed << " " << std::right << std::setw(10)
                                     << p.timesExcept << " " << p.name << "";
      }

      /*
      std::vector<int>::const_iterator epi = empty_trig_paths_.begin();
      std::vector<int>::const_iterator epe = empty_trig_paths_.end();
      std::vector<std::string>::const_iterator  epn = empty_trig_path_names_.begin();
      for (; epi != epe; ++epi, ++epn) {

        LogFwkVerbatim("FwkSummary") << "TrigReport "
        << std::right << std::setw(5) << 1
        << std::right << std::setw(5) << *epi << " "
        << std::right << std::setw(10) << totalEvents() << " "
        << std::right << std::setw(10) << totalEvents() << " "
        << std::right << std::setw(10) << 0 << " "
        << std::right << std::setw(10) << 0 << " "
        << *epn << "";
      }
       */

      LogFwkVerbatim("FwkSummary") << "";
      LogFwkVerbatim("FwkSummary") << "TrigReport "
                                   << "-------End-Path   Summary ------------";
      LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << "Trig Bit#"
                                   << " " << std::right << std::setw(10) << "Executed"
                                   << " " << std::right << std::setw(10) << "Passed"
                                   << " " << std::right << std::setw(10) << "Failed"
                                   << " " << std::right << std::setw(10) << "Error"
                                   << " "
                                   << "Name"
                                   << "";
      for (auto const& p : tr.endPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(5) << 0 << std::right << std::setw(5)
                                     << p.bitPosition << " " << std::right << std::setw(10) << p.timesRun << " "
                                     << std::right << std::setw(10) << p.timesPassed << " " << std::right
                                     << std::setw(10) << p.timesFailed << " " << std::right << std::setw(10)
                                     << p.timesExcept << " " << p.name << "";
      }

      for (auto const& p : tr.trigPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "";
        LogFwkVerbatim("FwkSummary") << "TrigReport "
                                     << "---------- Modules in Path: " << p.name << " ------------";
        LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << "Trig Bit#"
                                     << " " << std::right << std::setw(10) << "Visited"
                                     << " " << std::right << std::setw(10) << "Passed"
                                     << " " << std::right << std::setw(10) << "Failed"
                                     << " " << std::right << std::setw(10) << "Error"
                                     << " "
                                     << "Name"
                                     << "";

        for (auto const& mod : p.moduleInPathSummaries) {
          LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(5) << 1 << std::right << std::setw(5)
                                       << mod.bitPosition << " " << std::right << std::setw(10) << mod.timesVisited
                                       << " " << std::right << std::setw(10) << mod.timesPassed << " " << std::right
                                       << std::setw(10) << mod.timesFailed << " " << std::right << std::setw(10)
                                       << mod.timesExcept << " " << mod.moduleLabel << "";
        }
      }

      for (auto const& p : tr.endPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "";
        LogFwkVerbatim("FwkSummary") << "TrigReport "
                                     << "------ Modules in End-Path: " << p.name << " ------------";
        LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << "Trig Bit#"
                                     << " " << std::right << std::setw(10) << "Visited"
                                     << " " << std::right << std::setw(10) << "Passed"
                                     << " " << std::right << std::setw(10) << "Failed"
                                     << " " << std::right << std::setw(10) << "Error"
                                     << " "
                                     << "Name"
                                     << "";

        unsigned int bitpos = 0;
        for (auto const& mod : p.moduleInPathSummaries) {
          LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(5) << 0 << std::right << std::setw(5)
                                       << bitpos << " " << std::right << std::setw(10) << mod.timesVisited << " "
                                       << std::right << std::setw(10) << mod.timesPassed << " " << std::right
                                       << std::setw(10) << mod.timesFailed << " " << std::right << std::setw(10)
                                       << mod.timesExcept << " " << mod.moduleLabel << "";
          ++bitpos;
        }
      }

      LogFwkVerbatim("FwkSummary") << "";
      LogFwkVerbatim("FwkSummary") << "TrigReport "
                                   << "---------- Module Summary ------------";
      LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << "Visited"
                                   << " " << std::right << std::setw(10) << "Executed"
                                   << " " << std::right << std::setw(10) << "Passed"
                                   << " " << std::right << std::setw(10) << "Failed"
                                   << " " << std::right << std::setw(10) << "Error"
                                   << " "
                                   << "Name"
                                   << "";
      for (auto const& worker : tr.workerSummaries) {
        LogFwkVerbatim("FwkSummary") << "TrigReport " << std::right << std::setw(10) << worker.timesVisited << " "
                                     << std::right << std::setw(10) << worker.timesRun << " " << std::right
                                     << std::setw(10) << worker.timesPassed << " " << std::right << std::setw(10)
                                     << worker.timesFailed << " " << std::right << std::setw(10) << worker.timesExcept
                                     << " " << worker.moduleLabel << "";
      }
      LogFwkVerbatim("FwkSummary") << "";
    }
    // The timing report (CPU and Real Time):
    TriggerTimingReport tr;
    getTriggerTimingReport(tr);

    const int totalEvents = std::max(1, tr.eventSummary.totalEvents);

    LogFwkVerbatim("FwkSummary") << "TimeReport "
                                 << "---------- Event  Summary ---[sec]----";
    LogFwkVerbatim("FwkSummary") << "TimeReport" << std::setprecision(6) << std::fixed
                                 << "       event loop CPU/event = " << tr.eventSummary.cpuTime / totalEvents;
    LogFwkVerbatim("FwkSummary") << "TimeReport" << std::setprecision(6) << std::fixed
                                 << "      event loop Real/event = " << tr.eventSummary.realTime / totalEvents;
    LogFwkVerbatim("FwkSummary") << "TimeReport" << std::setprecision(6) << std::fixed
                                 << "     sum Streams Real/event = " << tr.eventSummary.sumStreamRealTime / totalEvents;
    LogFwkVerbatim("FwkSummary") << "TimeReport" << std::setprecision(6) << std::fixed
                                 << " efficiency CPU/Real/thread = "
                                 << tr.eventSummary.cpuTime / tr.eventSummary.realTime /
                                        preallocConfig_.numberOfThreads();

    constexpr int kColumn1Size = 10;
    constexpr int kColumn2Size = 12;
    constexpr int kColumn3Size = 12;
    LogFwkVerbatim("FwkSummary") << "";
    LogFwkVerbatim("FwkSummary") << "TimeReport "
                                 << "---------- Path   Summary ---[Real sec]----";
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << "  Name";
    for (auto const& p : tr.trigPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::setprecision(6) << std::fixed << std::right
                                   << std::setw(kColumn1Size) << p.realTime / totalEvents << " " << std::right
                                   << std::setw(kColumn2Size) << p.realTime / timesRun << "  " << p.name << "";
    }
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << "  Name"
                                 << "";

    LogFwkVerbatim("FwkSummary") << "";
    LogFwkVerbatim("FwkSummary") << "TimeReport "
                                 << "-------End-Path   Summary ---[Real sec]----";
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << "  Name"
                                 << "";
    for (auto const& p : tr.endPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);

      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::setprecision(6) << std::fixed << std::right
                                   << std::setw(kColumn1Size) << p.realTime / totalEvents << " " << std::right
                                   << std::setw(kColumn2Size) << p.realTime / timesRun << "  " << p.name << "";
    }
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << "  Name"
                                 << "";

    for (auto const& p : tr.trigPathSummaries) {
      LogFwkVerbatim("FwkSummary") << "";
      LogFwkVerbatim("FwkSummary") << "TimeReport "
                                   << "---------- Modules in Path: " << p.name << " ---[Real sec]----";
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                   << " " << std::right << std::setw(kColumn2Size) << "per visit"
                                   << "  Name"
                                   << "";
      for (auto const& mod : p.moduleInPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "TimeReport " << std::setprecision(6) << std::fixed << std::right
                                     << std::setw(kColumn1Size) << mod.realTime / totalEvents << " " << std::right
                                     << std::setw(kColumn2Size) << mod.realTime / std::max(1, mod.timesVisited) << "  "
                                     << mod.moduleLabel << "";
      }
    }
    if (not tr.trigPathSummaries.empty()) {
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                   << " " << std::right << std::setw(kColumn2Size) << "per visit"
                                   << "  Name"
                                   << "";
    }
    for (auto const& p : tr.endPathSummaries) {
      LogFwkVerbatim("FwkSummary") << "";
      LogFwkVerbatim("FwkSummary") << "TimeReport "
                                   << "------ Modules in End-Path: " << p.name << " ---[Real sec]----";
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                   << " " << std::right << std::setw(kColumn2Size) << "per visit"
                                   << "  Name"
                                   << "";
      for (auto const& mod : p.moduleInPathSummaries) {
        LogFwkVerbatim("FwkSummary") << "TimeReport " << std::setprecision(6) << std::fixed << std::right
                                     << std::setw(kColumn1Size) << mod.realTime / totalEvents << " " << std::right
                                     << std::setw(kColumn2Size) << mod.realTime / std::max(1, mod.timesVisited) << "  "
                                     << mod.moduleLabel << "";
      }
    }
    if (not tr.endPathSummaries.empty()) {
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                   << " " << std::right << std::setw(kColumn2Size) << "per visit"
                                   << "  Name"
                                   << "";
    }
    LogFwkVerbatim("FwkSummary") << "";
    LogFwkVerbatim("FwkSummary") << "TimeReport "
                                 << "---------- Module Summary ---[Real sec]----";
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << " " << std::right << std::setw(kColumn3Size) << "per visit"
                                 << "  Name"
                                 << "";
    for (auto const& worker : tr.workerSummaries) {
      LogFwkVerbatim("FwkSummary") << "TimeReport " << std::setprecision(6) << std::fixed << std::right
                                   << std::setw(kColumn1Size) << worker.realTime / totalEvents << " " << std::right
                                   << std::setw(kColumn2Size) << worker.realTime / std::max(1, worker.timesRun) << " "
                                   << std::right << std::setw(kColumn3Size)
                                   << worker.realTime / std::max(1, worker.timesVisited) << "  " << worker.moduleLabel
                                   << "";
    }
    LogFwkVerbatim("FwkSummary") << "TimeReport " << std::right << std::setw(kColumn1Size) << "per event"
                                 << " " << std::right << std::setw(kColumn2Size) << "per exec"
                                 << " " << std::right << std::setw(kColumn3Size) << "per visit"
                                 << "  Name"
                                 << "";

    LogFwkVerbatim("FwkSummary") << "";
    LogFwkVerbatim("FwkSummary") << "T---Report end!"
                                 << "";
    LogFwkVerbatim("FwkSummary") << "";
  }

  void Schedule::closeOutputFiles() {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::closeFile, _1));
    for (auto& worker : allWorkers()) {
      worker->respondToCloseOutputFile();
    }
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
    auto token = ServiceRegistry::instance().presentToken();
    GlobalContext globalContext(GlobalContext::Transition::kWriteRun,
                                LuminosityBlockID(rp.run(), 0),
                                rp.index(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                rp.endTime(),
                                processContext);

    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      //services can depend on other services
      ServiceRegistry::Operate op(token);

      // Propagating the exception would be nontrivial, and signal actions are not supposed to throw exceptions
      CMS_SA_ALLOW try { activityRegistry->preGlobalWriteRunSignal_(globalContext); } catch (...) {
      }
      for (auto& c : all_output_communicators_) {
        c->writeRunAsync(nextTask, rp, processContext, activityRegistry, mergeableRunProductMetadata);
      }
    }) | chain::then(doCleanup([activityRegistry, globalContext, token]() {
      //services can depend on other services
      ServiceRegistry::Operate op(token);

      activityRegistry->postGlobalWriteRunSignal_(globalContext);
    })) |
        chain::runLast(task);
  }

  void Schedule::writeProcessBlockAsync(WaitingTaskHolder task,
                                        ProcessBlockPrincipal const& pbp,
                                        ProcessContext const* processContext,
                                        ActivityRegistry* activityRegistry) {
    auto token = ServiceRegistry::instance().presentToken();
    GlobalContext globalContext(GlobalContext::Transition::kWriteProcessBlock,
                                LuminosityBlockID(),
                                RunIndex::invalidRunIndex(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                Timestamp::invalidTimestamp(),
                                processContext);

    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      // Propagating the exception would be nontrivial, and signal actions are not supposed to throw exceptions
      ServiceRegistry::Operate op(token);
      CMS_SA_ALLOW try { activityRegistry->preWriteProcessBlockSignal_(globalContext); } catch (...) {
      }
      for (auto& c : all_output_communicators_) {
        c->writeProcessBlockAsync(nextTask, pbp, processContext, activityRegistry);
      }
    }) | chain::then(doCleanup([activityRegistry, globalContext, token]() {
      //services can depend on other services
      ServiceRegistry::Operate op(token);

      activityRegistry->postWriteProcessBlockSignal_(globalContext);
    })) |
        chain::runLast(std::move(task));
  }

  void Schedule::writeLumiAsync(WaitingTaskHolder task,
                                LuminosityBlockPrincipal const& lbp,
                                ProcessContext const* processContext,
                                ActivityRegistry* activityRegistry) {
    auto token = ServiceRegistry::instance().presentToken();
    GlobalContext globalContext(GlobalContext::Transition::kWriteLuminosityBlock,
                                lbp.id(),
                                lbp.runPrincipal().index(),
                                lbp.index(),
                                lbp.beginTime(),
                                processContext);

    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      ServiceRegistry::Operate op(token);
      CMS_SA_ALLOW try { activityRegistry->preGlobalWriteLumiSignal_(globalContext); } catch (...) {
      }
      for (auto& c : all_output_communicators_) {
        c->writeLumiAsync(nextTask, lbp, processContext, activityRegistry);
      }
    }) | chain::then(doCleanup([activityRegistry, globalContext, token]() {
      //services can depend on other services
      ServiceRegistry::Operate op(token);

      activityRegistry->postGlobalWriteLumiSignal_(globalContext);
    })) |
        chain::runLast(task);
  }

  bool Schedule::shouldWeCloseOutput() const {
    using std::placeholders::_1;
    // Return true iff at least one output module returns true.
    return (std::find_if(all_output_communicators_.begin(),
                         all_output_communicators_.end(),
                         std::bind(&OutputModuleCommunicator::shouldWeCloseFile, _1)) !=
            all_output_communicators_.end());
  }

  void Schedule::respondToOpenInputFile(FileBlock const& fb) {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::respondToOpenInputFile, _1, std::cref(fb)));
  }

  void Schedule::respondToCloseInputFile(FileBlock const& fb) {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::respondToCloseInputFile, _1, std::cref(fb)));
  }

  void Schedule::beginJob(ProductRegistry const& iRegistry,
                          eventsetup::ESRecordsToProxyIndices const& iESIndices,
                          ProcessBlockHelperBase const& processBlockHelperBase) {
    globalSchedule_->beginJob(iRegistry, iESIndices, processBlockHelperBase);
  }

  void Schedule::beginStream(unsigned int iStreamID) {
    assert(iStreamID < streamSchedules_.size());
    streamSchedules_[iStreamID]->beginStream();
  }

  void Schedule::endStream(unsigned int iStreamID) {
    assert(iStreamID < streamSchedules_.size());
    streamSchedules_[iStreamID]->endStream();
  }

  void Schedule::processOneEventAsync(WaitingTaskHolder iTask,
                                      unsigned int iStreamID,
                                      EventTransitionInfo& info,
                                      ServiceToken const& token) {
    assert(iStreamID < streamSchedules_.size());
    streamSchedules_[iStreamID]->processOneEventAsync(std::move(iTask), info, token, pathStatusInserters_);
  }

  bool Schedule::changeModule(std::string const& iLabel,
                              ParameterSet const& iPSet,
                              const ProductRegistry& iRegistry,
                              eventsetup::ESRecordsToProxyIndices const& iIndices) {
    Worker* found = nullptr;
    for (auto const& worker : allWorkers()) {
      if (worker->description()->moduleLabel() == iLabel) {
        found = worker;
        break;
      }
    }
    if (nullptr == found) {
      return false;
    }

    auto newMod = moduleRegistry_->replaceModule(iLabel, iPSet, preallocConfig_);

    globalSchedule_->replaceModule(newMod, iLabel);

    for (auto& s : streamSchedules_) {
      s->replaceModule(newMod, iLabel);
    }

    {
      //Need to updateLookup in order to make getByToken work
      auto const processBlockLookup = iRegistry.productLookup(InProcess);
      auto const runLookup = iRegistry.productLookup(InRun);
      auto const lumiLookup = iRegistry.productLookup(InLumi);
      auto const eventLookup = iRegistry.productLookup(InEvent);
      found->updateLookup(InProcess, *runLookup);
      found->updateLookup(InRun, *runLookup);
      found->updateLookup(InLumi, *lumiLookup);
      found->updateLookup(InEvent, *eventLookup);
      found->updateLookup(iIndices);

      auto const& processName = newMod->moduleDescription().processName();
      auto const& processBlockModuleToIndicies = processBlockLookup->indiciesForModulesInProcess(processName);
      auto const& runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
      auto const& lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
      auto const& eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);
      found->resolvePutIndicies(InProcess, processBlockModuleToIndicies);
      found->resolvePutIndicies(InRun, runModuleToIndicies);
      found->resolvePutIndicies(InLumi, lumiModuleToIndicies);
      found->resolvePutIndicies(InEvent, eventModuleToIndicies);
    }

    return true;
  }

  void Schedule::deleteModule(std::string const& iLabel, ActivityRegistry* areg) {
    globalSchedule_->deleteModule(iLabel);
    for (auto& stream : streamSchedules_) {
      stream->deleteModule(iLabel);
    }
    moduleRegistry_->deleteModule(iLabel, areg->preModuleDestructionSignal_, areg->postModuleDestructionSignal_);
  }

  void Schedule::initializeEarlyDelete(std::vector<std::string> const& branchesToDeleteEarly,
                                       std::multimap<std::string, std::string> const& referencesToBranches,
                                       std::vector<std::string> const& modulesToSkip,
                                       edm::ProductRegistry const& preg) {
    for (auto& stream : streamSchedules_) {
      stream->initializeEarlyDelete(
          *moduleRegistry(), branchesToDeleteEarly, referencesToBranches, modulesToSkip, preg);
    }
  }

  std::vector<ModuleDescription const*> Schedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->description();
      result.push_back(p);
    }
    return result;
  }

  Schedule::AllWorkers const& Schedule::allWorkers() const { return globalSchedule_->allWorkers(); }

  void Schedule::convertCurrentProcessAlias(std::string const& processName) {
    for (auto const& worker : allWorkers()) {
      worker->convertCurrentProcessAlias(processName);
    }
  }

  void Schedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->availablePaths(oLabelsToFill);
  }

  void Schedule::triggerPaths(std::vector<std::string>& oLabelsToFill) const { oLabelsToFill = *pathNames_; }

  void Schedule::endPaths(std::vector<std::string>& oLabelsToFill) const { oLabelsToFill = *endPathNames_; }

  void Schedule::modulesInPath(std::string const& iPathLabel, std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->modulesInPath(iPathLabel, oLabelsToFill);
  }

  void Schedule::moduleDescriptionsInPath(std::string const& iPathLabel,
                                          std::vector<ModuleDescription const*>& descriptions,
                                          unsigned int hint) const {
    streamSchedules_[0]->moduleDescriptionsInPath(iPathLabel, descriptions, hint);
  }

  void Schedule::moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                             std::vector<ModuleDescription const*>& descriptions,
                                             unsigned int hint) const {
    streamSchedules_[0]->moduleDescriptionsInEndPath(iEndPathLabel, descriptions, hint);
  }

  void Schedule::fillModuleAndConsumesInfo(
      std::vector<ModuleDescription const*>& allModuleDescriptions,
      std::vector<std::pair<unsigned int, unsigned int>>& moduleIDToIndex,
      std::array<std::vector<std::vector<ModuleDescription const*>>, NumBranchTypes>& modulesWhoseProductsAreConsumedBy,
      std::vector<std::vector<ModuleProcessName>>& modulesInPreviousProcessesWhoseProductsAreConsumedBy,
      ProductRegistry const& preg) const {
    allModuleDescriptions.clear();
    moduleIDToIndex.clear();
    for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
      modulesWhoseProductsAreConsumedBy[iBranchType].clear();
    }
    modulesInPreviousProcessesWhoseProductsAreConsumedBy.clear();

    allModuleDescriptions.reserve(allWorkers().size());
    moduleIDToIndex.reserve(allWorkers().size());
    for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
      modulesWhoseProductsAreConsumedBy[iBranchType].resize(allWorkers().size());
    }
    modulesInPreviousProcessesWhoseProductsAreConsumedBy.resize(allWorkers().size());

    std::map<std::string, ModuleDescription const*> labelToDesc;
    unsigned int i = 0;
    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->description();
      allModuleDescriptions.push_back(p);
      moduleIDToIndex.push_back(std::pair<unsigned int, unsigned int>(p->id(), i));
      labelToDesc[p->moduleLabel()] = p;
      ++i;
    }
    sort_all(moduleIDToIndex);

    i = 0;
    for (auto const& worker : allWorkers()) {
      std::array<std::vector<ModuleDescription const*>*, NumBranchTypes> modules;
      for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
        modules[iBranchType] = &modulesWhoseProductsAreConsumedBy[iBranchType].at(i);
      }

      std::vector<ModuleProcessName>& modulesInPreviousProcesses =
          modulesInPreviousProcessesWhoseProductsAreConsumedBy.at(i);
      try {
        worker->modulesWhoseProductsAreConsumed(modules, modulesInPreviousProcesses, preg, labelToDesc);
      } catch (cms::Exception& ex) {
        ex.addContext("Calling Worker::modulesWhoseProductsAreConsumed() for module " +
                      worker->description()->moduleLabel());
        throw;
      }
      ++i;
    }
  }

  void Schedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents = 0;
    rep.eventSummary.totalEventsPassed = 0;
    rep.eventSummary.totalEventsFailed = 0;
    for (auto& s : streamSchedules_) {
      s->getTriggerReport(rep);
    }
    sort_all(rep.workerSummaries);
  }

  void Schedule::getTriggerTimingReport(TriggerTimingReport& rep) const {
    rep.eventSummary.totalEvents = 0;
    rep.eventSummary.cpuTime = 0.;
    rep.eventSummary.realTime = 0.;
    summaryTimeKeeper_->fillTriggerTimingReport(rep);
  }

  int Schedule::totalEvents() const {
    int returnValue = 0;
    for (auto& s : streamSchedules_) {
      returnValue += s->totalEvents();
    }
    return returnValue;
  }

  int Schedule::totalEventsPassed() const {
    int returnValue = 0;
    for (auto& s : streamSchedules_) {
      returnValue += s->totalEventsPassed();
    }
    return returnValue;
  }

  int Schedule::totalEventsFailed() const {
    int returnValue = 0;
    for (auto& s : streamSchedules_) {
      returnValue += s->totalEventsFailed();
    }
    return returnValue;
  }

  void Schedule::clearCounters() {
    for (auto& s : streamSchedules_) {
      s->clearCounters();
    }
  }
}  // namespace edm
