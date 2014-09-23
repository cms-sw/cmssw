#include "FWCore/Framework/interface/Schedule.h"

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/interface/TriggerTimingReport.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/ModuleRegistry.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"

#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/visitors.hpp"


#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <list>
#include <map>
#include <exception>
#include <sstream>

namespace edm {
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
        maker::ModuleHolderT<TriggerResultInserter> holder(new TriggerResultInserter(*trig_pset, iPrealloc.numberOfStreams()),static_cast<Maker const*>(nullptr));
        holder.setModuleDescription(md);
        holder.registerProductsAndCallbacks(&preg);
        returnValue.reset(holder.release());
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
  }
  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  Schedule::Schedule(ParameterSet& proc_pset,
                     service::TriggerNamesService& tns,
                     ProductRegistry& preg,
                     BranchIDListHelper& branchIDListHelper,
                     ThinnedAssociationsHelper& thinnedAssociationsHelper,
                     ExceptionToActionTable const& actions,
                     std::shared_ptr<ActivityRegistry> areg,
                     std::shared_ptr<ProcessConfiguration> processConfiguration,
                     const ParameterSet* subProcPSet,
                     PreallocationConfiguration const& prealloc,
                     ProcessContext const* processContext) :
  //Only create a resultsInserter if there is a trigger path
  resultsInserter_{tns.getTrigPaths().empty()? std::shared_ptr<TriggerResultInserter>{} :makeInserter(proc_pset,prealloc,preg,actions,areg,processConfiguration)},
    moduleRegistry_(new ModuleRegistry()),
    all_output_communicators_(),
    preallocConfig_(prealloc),
    wantSummary_(tns.wantSummary()),
    endpathsAreActive_(true)
  {
    assert(0<prealloc.numberOfStreams());
    streamSchedules_.reserve(prealloc.numberOfStreams());
    for(unsigned int i=0; i<prealloc.numberOfStreams();++i) {
      streamSchedules_.emplace_back(std::make_shared<StreamSchedule>(resultsInserter_.get(),moduleRegistry_,proc_pset,tns,prealloc,preg,branchIDListHelper,actions,areg,processConfiguration,nullptr==subProcPSet,StreamID{i},processContext));
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
    globalSchedule_.reset( new GlobalSchedule{ resultsInserter_.get(),
      moduleRegistry_,
      modulesToUse,
      proc_pset, preg, prealloc,
      actions,areg,processConfiguration,processContext });
    
    //TriggerResults is not in the top level ParameterSet so the call to
    // reduceParameterSet would fail to find it. Just remove it up front.
    std::set<std::string> usedModuleLabels;
    for( auto const worker: allWorkers()) {
      if(worker->description().moduleLabel() != kTriggerResults) {
        usedModuleLabels.insert(worker->description().moduleLabel());
      }
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string> >("@all_modules"));
    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    reduceParameterSet(proc_pset, tns.getEndPaths(), modulesInConfig, usedModuleLabels,
                       outputModulePathPositions);
    processEDAliases(proc_pset, processConfiguration->processName(), preg);
    proc_pset.registerIt();
    pset::setProcessParameterSetID(proc_pset.id());
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
    limitOutput(proc_pset, branchIDListHelper.branchIDLists());

    loadMissingDictionaries();

    // Sanity check: make sure nobody has added a worker after we've
    // already relied on the WorkerManager being full.
    assert (all_workers_count == allWorkers().size());

    branchIDListHelper.updateFromRegistry(preg);

    preg.setFrozen();

    for(auto const& worker : streamSchedules_[0]->allWorkers()) {
      worker->registerThinnedAssociations(preg, thinnedAssociationsHelper);
    }
    thinnedAssociationsHelper.sort();

    for (auto c : all_output_communicators_) {
      c->setEventSelectionInfo(outputModulePathPositions, preg.anyProductProduced());
      c->selectProducts(preg, thinnedAssociationsHelper);
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
      
      summaryTimeKeeper_.reset(new SystemTimeKeeper(prealloc.numberOfStreams(),
                                                    modDesc,
                                                    tns));
      auto timeKeeperPtr = summaryTimeKeeper_.get();
      
      areg->watchPreModuleEvent(timeKeeperPtr, &SystemTimeKeeper::startModuleEvent);
      areg->watchPostModuleEvent(timeKeeperPtr, &SystemTimeKeeper::stopModuleEvent);
      areg->watchPreModuleEventDelayedGet(timeKeeperPtr, &SystemTimeKeeper::pauseModuleEvent);
      areg->watchPostModuleEventDelayedGet(timeKeeperPtr,&SystemTimeKeeper::restartModuleEvent);
      
      areg->watchPreSourceEvent(timeKeeperPtr, &SystemTimeKeeper::startEvent);
      areg->watchPostEvent(timeKeeperPtr, &SystemTimeKeeper::stopEvent);
      
      areg->watchPrePathEvent(timeKeeperPtr, &SystemTimeKeeper::startPath);
      areg->watchPostPathEvent(timeKeeperPtr, &SystemTimeKeeper::stopPath);
      //areg->preModuleEventSignal_.connect([timeKeeperPtr](StreamContext const& iContext, ModuleCallingContext const& iMod) {
      //timeKeeperPtr->startModuleEvent(iContext,iMod);
      //});
    }

  } // Schedule::Schedule

  
  void
  Schedule::limitOutput(ParameterSet const& proc_pset, BranchIDLists const& branchIDLists) {
    std::string const output("output");

    ParameterSet const& maxEventsPSet = proc_pset.getUntrackedParameterSet("maxEvents", ParameterSet());
    int maxEventSpecs = 0;
    int maxEventsOut = -1;
    ParameterSet const* vMaxEventsOut = 0;
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

    for (auto c : all_output_communicators_) {
      OutputModuleDescription desc(branchIDLists, maxEventsOut);
      if (vMaxEventsOut != 0 && !vMaxEventsOut->empty()) {
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
    for (auto c : all_output_communicators_) {
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
      << std::right << std::setw(10) << "Run" << " "
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
      << std::right << std::setw(10) << "Run" << " "
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
      << std::right << std::setw(10) << "Run" << " "
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
                              << " CPU/event = " << tr.eventSummary.cpuTime/totalEvents
                              << " Real/event = " << tr.eventSummary.realTime/totalEvents
                              << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Path   Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per path-run "
                              << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    for (auto const& p: tr.trigPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << p.cpuTime/totalEvents << " "
                                << std::right << std::setw(10) << p.realTime/totalEvents << " "
                                << std::right << std::setw(10) << p.cpuTime/timesRun << " "
                                << std::right << std::setw(10) << p.realTime/timesRun << " "
                                << p.name << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per path-run "
                              << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "-------End-Path   Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per endpath-run "
                              << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    for (auto const& p: tr.endPathSummaries) {
      const int timesRun = std::max(1, p.timesRun);

      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << p.cpuTime/totalEvents << " "
                                << std::right << std::setw(10) << p.realTime/totalEvents << " "
                                << std::right << std::setw(10) << p.cpuTime/timesRun << " "
                                << std::right << std::setw(10) << p.realTime/timesRun << " "
                                << p.name << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per endpath-run "
                              << "";

    for (auto const& p: tr.trigPathSummaries) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "---------- Modules in Path: " << p.name << " ---[sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(22) << "per event "
                                << std::right << std::setw(22) << "per module-visit "
                                << "";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(10) << "CPU" << " "
                                << std::right << std::setw(10) << "Real" << " "
                                << std::right << std::setw(10) << "CPU" << " "
                                << std::right << std::setw(10) << "Real" << " "
                                << "Name" << "";
      for (auto const& mod: p.moduleInPathSummaries) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(10) << mod.cpuTime/totalEvents << " "
                                  << std::right << std::setw(10) << mod.realTime/totalEvents << " "
                                  << std::right << std::setw(10) << mod.cpuTime/std::max(1, mod.timesVisited) << " "
                                  << std::right << std::setw(10) << mod.realTime/std::max(1, mod.timesVisited) << " "
                                  << mod.moduleLabel << "";
      }
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per module-visit "
                              << "";

    for (auto const& p: tr.endPathSummaries) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "------ Modules in End-Path: " << p.name << " ---[sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(22) << "per event "
                                << std::right << std::setw(22) << "per module-visit "
                                << "";
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::right << std::setw(10) << "CPU" << " "
                                << std::right << std::setw(10) << "Real" << " "
                                << std::right << std::setw(10) << "CPU" << " "
                                << std::right << std::setw(10) << "Real" << " "
                                << "Name" << "";
      for (auto const& mod: p.moduleInPathSummaries) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(10) << mod.cpuTime/totalEvents << " "
                                  << std::right << std::setw(10) << mod.realTime/totalEvents << " "
                                  << std::right << std::setw(10) << mod.cpuTime/std::max(1, mod.timesVisited) << " "
                                  << std::right << std::setw(10) << mod.realTime/std::max(1, mod.timesVisited) << " "
                                  << mod.moduleLabel << "";
      }
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per module-visit "
                              << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Module Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per module-run "
                              << std::right << std::setw(22) << "per module-visit "
                              << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    for (auto const& worker : tr.workerSummaries) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << worker.cpuTime/totalEvents << " "
                                << std::right << std::setw(10) << worker.realTime/totalEvents << " "
                                << std::right << std::setw(10) << worker.cpuTime/std::max(1, worker.timesRun) << " "
                                << std::right << std::setw(10) << worker.realTime/std::max(1, worker.timesRun) << " "
                                << std::right << std::setw(10) << worker.cpuTime/std::max(1, worker.timesVisited) << " "
                                << std::right << std::setw(10) << worker.realTime/std::max(1, worker.timesVisited) << " "
                                << worker.moduleLabel << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << std::right << std::setw(10) << "CPU" << " "
                              << std::right << std::setw(10) << "Real" << " "
                              << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
                              << std::right << std::setw(22) << "per event "
                              << std::right << std::setw(22) << "per module-run "
                              << std::right << std::setw(22) << "per module-visit "
                              << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "T---Report end!" << "";
    LogVerbatim("FwkSummary") << "";
  }

  void Schedule::closeOutputFiles() {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::closeFile, _1));
  }

  void Schedule::openNewOutputFilesIfNeeded() {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::openNewFileIfNeeded, _1));
  }

  void Schedule::openOutputFiles(FileBlock& fb) {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::openFile, _1, std::cref(fb)));
  }

  void Schedule::writeRun(RunPrincipal const& rp, ProcessContext const* processContext) {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::writeRun, _1, std::cref(rp), processContext));
  }

  void Schedule::writeLumi(LuminosityBlockPrincipal const& lbp, ProcessContext const* processContext) {
    using std::placeholders::_1;
    for_all(all_output_communicators_, std::bind(&OutputModuleCommunicator::writeLumi, _1, std::cref(lbp), processContext));
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
    checkForCorrectness();
    
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

  void Schedule::preForkReleaseResources() {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::preForkReleaseResources, _1));
  }
  void Schedule::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    using std::placeholders::_1;
    for_all(allWorkers(), std::bind(&Worker::postForkReacquireResources, _1, iChildIndex, iNumberOfChildren));
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

    for(auto s: streamSchedules_) {
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
  
  void
  Schedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->availablePaths(oLabelsToFill);
  }

  void
  Schedule::modulesInPath(std::string const& iPathLabel,
                          std::vector<std::string>& oLabelsToFill) const {
    streamSchedules_[0]->modulesInPath(iPathLabel,oLabelsToFill);
  }

  void
  Schedule::enableEndPaths(bool active) {
    endpathsAreActive_ = active;
    for(auto const &  s : streamSchedules_) {
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
    for(auto const& s: streamSchedules_) {
      s->clearCounters();
    }
  }
  
  //====================================
  // Schedule::checkForCorrectness algorithm
  //
  // The code creates a 'dependency' graph between all
  // modules. A module depends on another module if
  // 1) it 'consumes' data produced by that module
  // 2) it appears directly after the module within a Path
  //
  // If there is a cycle in the 'dependency' graph then
  // the schedule may be unrunnable. The schedule is still
  // runnable if all cycles have at least two edges which
  // connect modules only by Path dependencies (i.e. not
  // linked by a data dependency).
  //
  //  Example 1:
  //  C consumes data from B
  //  Path 1: A + B + C
  //  Path 2: B + C + A
  //
  //  Cycle: A after C [p2], C consumes B, B after A [p1]
  //  Since this cycle has 2 path only edges it is OK since
  //  A and (B+C) are independent so their run order doesn't matter
  //
  //  Example 2:
  //  B consumes A
  //  C consumes B
  //  Path: C + A
  //
  //  Cycle: A after C [p], C consumes B, B consumes A
  //  Since this cycle has 1 path only edge it is unrunnable.
  //
  //  Example 3:
  //  A consumes B
  //  B consumes C
  //  C consumes A
  //  (no Path since unscheduled execution)
  //
  //  Cycle: A consumes B, B consumes C, C consumes A
  //  Since this cycle has 0 path only edges it is unrunnable.
  //====================================
  
  namespace {
    typedef std::pair<unsigned int, unsigned int> SimpleEdge;
    typedef std::map<SimpleEdge, std::vector<unsigned int>> EdgeToPathMap;
    
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;

    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    struct cycle_detector : public boost::dfs_visitor<> {
      
      cycle_detector(EdgeToPathMap const& iEdgeToPathMap,
                     std::vector<std::string> const& iPathNames,
                     std::map<std::string,unsigned int> const& iModuleNamesToIndex):
      m_edgeToPathMap(iEdgeToPathMap),
      m_pathNames(iPathNames),
      m_namesToIndex(iModuleNamesToIndex){}
      
      void tree_edge(Edge iEdge, Graph const&) {
        m_stack.push_back(iEdge);
      }

      void finish_edge(Edge iEdge, Graph const& iGraph) {
        if(not m_stack.empty()) {
          if (iEdge == m_stack.back()) {
            m_stack.pop_back();
          }
        }
      }

      //Called if a cycle happens
      void back_edge(Edge iEdge, Graph const& iGraph) {
        //NOTE: If the path containing the cycle contains two or more
        // path only edges then there is no problem
        
        typedef typename boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
        IndexMap const& index = get(boost::vertex_index, iGraph);
        
        unsigned int vertex = index[target(iEdge,iGraph)];

        //Find last edge which starts with this vertex
        std::list<Edge>::iterator itFirst = m_stack.begin();
        {
          bool seenVertex = false;
          while(itFirst != m_stack.end()) {
            if(not seenVertex) {
              if(index[source(*itFirst,iGraph)] == vertex) {
                seenVertex = true;
              }
            } else
            if (index[source(*itFirst,iGraph)] != vertex) {
              break;
            }
            ++itFirst;
          }
          if(itFirst != m_stack.begin()) {
            --itFirst;
          }
        }
        //This edge has not been added to the stack yet
        // making a copy allows us to add it in but not worry
        // about removing it at the end of the routine
        std::vector<Edge> tempStack;
        tempStack.reserve(m_stack.size()+1);
        tempStack.insert(tempStack.end(),itFirst,m_stack.end());
        tempStack.emplace_back(iEdge);
        
        unsigned int nPathDependencyOnly =0;
        for(auto const& edge: tempStack) {
          unsigned int in =index[source(edge,iGraph)];
          unsigned int out =index[target(edge,iGraph)];

          auto iFound = m_edgeToPathMap.find(SimpleEdge(in,out));
          bool pathDependencyOnly = true;
          for(auto dependency : iFound->second) {
            if (dependency == std::numeric_limits<unsigned int>::max()) {
              pathDependencyOnly = false;
              break;
            }
          }
          if (pathDependencyOnly) {
            ++nPathDependencyOnly;
          }
        }
        if(nPathDependencyOnly < 2) {
          reportError(tempStack,index,iGraph);
        }
      }
    private:
      std::string const& pathName(unsigned int iIndex) const {
        return m_pathNames[iIndex];
      }
      
      std::string const& moduleName(unsigned int iIndex) const {
        for(auto const& item : m_namesToIndex) {
          if(item.second == iIndex) {
            return item.first;
          }
        }
        assert(false);
      }
      
      void
      reportError(std::vector<Edge>const& iEdges,
                  boost::property_map<Graph, boost::vertex_index_t>::type const& iIndex,
                  Graph const& iGraph) const {
        std::stringstream oStream;
        oStream <<"Module run order problem found: \n";
        bool first_edge = true;
        for(auto const& edge: iEdges) {
          unsigned int in =iIndex[source(edge,iGraph)];
          unsigned int out =iIndex[target(edge,iGraph)];
          
          if(first_edge) {
            first_edge = false;
          } else {
            oStream<<", ";
          }
          oStream <<moduleName(in);
          
          auto iFound = m_edgeToPathMap.find(SimpleEdge(in,out));
          bool pathDependencyOnly = true;
          for(auto dependency : iFound->second) {
            if (dependency == std::numeric_limits<unsigned int>::max()) {
              pathDependencyOnly = false;
              break;
            }
          }
          if (pathDependencyOnly) {
            oStream <<" after "<<moduleName(out)<<" [path "<<pathName(iFound->second[0])<<"]";
          } else {
            oStream <<" consumes "<<moduleName(out);
          }
        }
        oStream<<"\n Running in the threaded framework would lead to indeterminate results."
        "\n Please change order of modules in mentioned Path(s) to avoid inconsistent module ordering.";
        
        LogError("UnrunnableSchedule")<<oStream.str();
      }
      
      EdgeToPathMap const& m_edgeToPathMap;
      std::vector<std::string> const& m_pathNames;
      std::map<std::string,unsigned int> m_namesToIndex;
      
      std::list<Edge> m_stack;
    };
  }
  
  void
  Schedule::checkForCorrectness() const
  {
    //Need to lookup names to ids quickly
    std::map<std::string,unsigned int> moduleNamesToIndex;
    for(auto worker: allWorkers()) {
      moduleNamesToIndex.insert( std::make_pair(worker->description().moduleLabel(),
                                         worker->description().id()));
    }
    
    //If a module to module dependency comes from a path, remember which path
    EdgeToPathMap edgeToPathMap;

    //determine the path dependencies
    std::vector<std::string> pathNames;
    {
      streamSchedules_[0]->availablePaths(pathNames);
      
      std::vector<std::string> moduleNames;
      std::vector<std::string> reducedModuleNames;
      unsigned int pathIndex=0;
      for(auto const& path: pathNames) {
        moduleNames.clear();
        reducedModuleNames.clear();
        std::set<std::string> alreadySeenNames;
        
        streamSchedules_[0]->modulesInPath(path,moduleNames);
        std::string lastModuleName;
        unsigned int lastModuleIndex;
        for(auto const& name: moduleNames) {
          auto found = alreadySeenNames.insert(name);
          if(found.second) {
            //first time for this path
            unsigned int moduleIndex = moduleNamesToIndex[name];
            if(not lastModuleName.empty() ) {
              edgeToPathMap[std::make_pair(moduleIndex,lastModuleIndex)].push_back(pathIndex);
            }
            lastModuleName = name;
            lastModuleIndex = moduleIndex;
          }
        }
        ++pathIndex;
      }
    }
    {
      std::vector<const char*> dependentModules;
      //determine the data dependencies
      for(auto const& worker: allWorkers()) {
        dependentModules.clear();
        //NOTE: what about aliases?
        worker->modulesDependentUpon(dependentModules);
        auto found = moduleNamesToIndex.find(worker->description().moduleLabel());
        if (found == moduleNamesToIndex.end()) {
          //The module was from a previous process
          continue;
        }
        unsigned int moduleIndex = found->second;
        for(auto name: dependentModules) {
          edgeToPathMap[std::make_pair(moduleIndex, moduleNamesToIndex[name])].push_back(std::numeric_limits<unsigned int>::max());
        }
      }
    }
    //Now use boost graph library to find cycles in the dependencies
    std::vector<SimpleEdge> outList;
    outList.reserve(edgeToPathMap.size());
    for(auto const& edgeInfo: edgeToPathMap) {
      outList.push_back(edgeInfo.first);
    }
    
    Graph g(outList.begin(),outList.end(), moduleNamesToIndex.size());
    
    cycle_detector detector(edgeToPathMap,pathNames,moduleNamesToIndex);
    boost::depth_first_search(g,boost::visitor(detector));
  }
}
