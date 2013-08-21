#include "FWCore/Framework/interface/Schedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/interface/TriggerTimingReport.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"

#include "boost/bind.hpp"
#include "boost/ref.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <list>
#include <map>
#include <exception>

namespace edm {
  namespace {

    // Function template to transform each element in the input range to
    // a value placed into the output range. The supplied function
    // should take a const_reference to the 'input', and write to a
    // reference to the 'output'.
    template <typename InputIterator, typename ForwardIterator, typename Func>
    void
    transform_into(InputIterator begin, InputIterator end,
                   ForwardIterator out, Func func) {
      for (; begin != end; ++begin, ++out) func(*begin, *out);
    }

    // Function template that takes a sequence 'from', a sequence
    // 'to', and a callable object 'func'. It and applies
    // transform_into to fill the 'to' sequence with the values
    // calcuated by the callable object, taking care to fill the
    // outupt only if all calls succeed.
    template <typename FROM, typename TO, typename FUNC>
    void
    fill_summary(FROM const& from, TO& to, FUNC func) {
      if(to.size()!=from.size()) {
        TO temp(from.size());
        transform_into(from.begin(), from.end(), temp.begin(), func);
        to.swap(temp);
      } else {
        transform_into(from.begin(), from.end(), to.begin(), func);
      }
    }

    // -----------------------------

    bool binary_search_string(std::vector<std::string> const& v, std::string const& s) {
      return std::binary_search(v.begin(), v.end(), s);
    }
    
    void
    initializeBranchToReadingWorker(ParameterSet const& opts,
                                    ProductRegistry const& preg,
                                    std::multimap<std::string,Worker*>& branchToReadingWorker)
    {
      // See if any data has been marked to be deleted early (removing any duplicates)
      auto vBranchesToDeleteEarly = opts.getUntrackedParameter<std::vector<std::string>>("canDeleteEarly",std::vector<std::string>());
      if(not vBranchesToDeleteEarly.empty()) {
        std::sort(vBranchesToDeleteEarly.begin(),vBranchesToDeleteEarly.end(),std::less<std::string>());
        vBranchesToDeleteEarly.erase(std::unique(vBranchesToDeleteEarly.begin(),vBranchesToDeleteEarly.end()),
                                     vBranchesToDeleteEarly.end());
        
        // Are the requested items in the product registry?
        auto allBranchNames = preg.allBranchNames();
        //the branch names all end with a period, which we do not want to compare with
        for(auto & b:allBranchNames) {
          b.resize(b.size()-1);
        }
        std::sort(allBranchNames.begin(),allBranchNames.end(),std::less<std::string>());
        std::vector<std::string> temp;
        temp.reserve(vBranchesToDeleteEarly.size());  
        
        std::set_intersection(vBranchesToDeleteEarly.begin(),vBranchesToDeleteEarly.end(),
                              allBranchNames.begin(),allBranchNames.end(),
                              std::back_inserter(temp));
        vBranchesToDeleteEarly.swap(temp);
        if(temp.size() != vBranchesToDeleteEarly.size()) {
          std::vector<std::string> missingProducts;
          std::set_difference(temp.begin(),temp.end(),
                              vBranchesToDeleteEarly.begin(),vBranchesToDeleteEarly.end(),
                              std::back_inserter(missingProducts));
          LogInfo l("MissingProductsForCanDeleteEarly");
          l<<"The following products in the 'canDeleteEarly' list are not available in this job and will be ignored.";
          for(auto const& n:missingProducts){
            l<<"\n "<<n;
          }
        }
        //set placeholder for the branch, we will remove the nullptr if a
        // module actually wants the branch.
        for(auto const& branch:vBranchesToDeleteEarly) {
          branchToReadingWorker.insert(std::make_pair(branch, static_cast<Worker*>(nullptr)));
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

      std::vector<std::string> labelsToBeDropped;
      labelsToBeDropped.reserve(modulesInConfigSet.size());
      std::set_difference(modulesInConfigSet.begin(),modulesInConfigSet.end(),
                          usedModuleLabels.begin(),usedModuleLabels.end(),
                          std::back_inserter(labelsToBeDropped));

      for (auto const& modLabel: usedModuleLabels) {
        edmType = proc_pset.getParameterSet(modLabel).getParameter<std::string>(moduleEdmType);
        if (edmType == outputModule) {
          outputModuleLabels.push_back(modLabel);
          labelsToBeDropped.push_back(modLabel);
        }
      }

      // drop the parameter sets used to configure the modules
      for_all(labelsToBeDropped, boost::bind(&ParameterSet::eraseOrSetUntrackedParameterSet, boost::ref(proc_pset), _1));
      
      // drop the labels from @all_modules
      vstring::iterator endAfterRemove = std::remove_if(modulesInConfig.begin(), modulesInConfig.end(), boost::bind(binary_search_string, boost::ref(labelsToBeDropped), _1));
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
      vstring scheduledPaths = proc_pset.getParameter<vstring>("@paths");
      endAfterRemove = std::remove_if(scheduledPaths.begin(), scheduledPaths.end(), boost::bind(binary_search_string, boost::ref(endPathsToBeDropped), _1));
      scheduledPaths.erase(endAfterRemove, scheduledPaths.end());
      proc_pset.addParameter<vstring>(std::string("@paths"), scheduledPaths);
      
      // remove empty end paths from @end_paths
      vstring scheduledEndPaths = proc_pset.getParameter<vstring>("@end_paths");
      endAfterRemove = std::remove_if(scheduledEndPaths.begin(), scheduledEndPaths.end(), boost::bind(binary_search_string, boost::ref(endPathsToBeDropped), _1));
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
                     ExceptionToActionTable const& actions,
                     boost::shared_ptr<ActivityRegistry> areg,
                     boost::shared_ptr<ProcessConfiguration> processConfiguration,
                     const ParameterSet* subProcPSet,
                     PreallocationConfiguration const& config,
                     ProcessContext const* processContext) :
    moduleRegistry_(makeModuleRegistry()),
    all_output_communicators_(),
    wantSummary_(tns.wantSummary()),
    endpathsAreActive_(true)
  {

    assert(0<config.numberOfStreams());
    streamSchedules_.reserve(config.numberOfStreams());
    for(unsigned int i=0; i<config.numberOfStreams();++i) {
      streamSchedules_.emplace_back(std::shared_ptr<StreamSchedule>{new StreamSchedule{moduleRegistry_,proc_pset,tns,preg,branchIDListHelper,actions,areg,processConfiguration,nullptr==subProcPSet,StreamID{i},processContext}});
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
    globalSchedule_.reset( new GlobalSchedule{ moduleRegistry_,
      modulesToUse,
      proc_pset, preg,
      actions,areg,processConfiguration,processContext });
    
      /*
    //See if all modules were used
    std::set<std::string> usedWorkerLabels;
    for (auto const& worker : allWorkers()) {
      usedWorkerLabels.insert(worker->description().moduleLabel());
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string> >("@all_modules"));
    std::set<std::string> modulesInConfigSet(modulesInConfig.begin(), modulesInConfig.end());
    std::vector<std::string> unusedLabels;
    set_difference(modulesInConfigSet.begin(), modulesInConfigSet.end(),
                   usedWorkerLabels.begin(), usedWorkerLabels.end(),
                   back_inserter(unusedLabels));
    //does the configuration say we should allow on demand?
    bool allowUnscheduled = opts.getUntrackedParameter<bool>("allowUnscheduled", false);
    std::set<std::string> unscheduledLabels;
    std::vector<std::string>  shouldBeUsedLabels;
    if (!unusedLabels.empty()) {
      //Need to
      // 1) create worker
      // 2) if it is a WorkerT<EDProducer>, add it to our list
      // 3) hand list to our delayed reader
      for (auto const& label : unusedLabels) {
        if (allowUnscheduled) {
          bool isTracked;
          ParameterSet* modulePSet(proc_pset.getPSetForUpdate(label, isTracked));
          assert(isTracked);
          assert(modulePSet != nullptr);
          workerManager_.addToUnscheduledWorkers(*modulePSet, preg, processConfiguration, label, wantSummary_, unscheduledLabels, shouldBeUsedLabels);
        } else {
          //everthing is marked are unused so no 'on demand' allowed
          shouldBeUsedLabels.push_back(label);
        }
      }
      if (!shouldBeUsedLabels.empty()) {
        std::ostringstream unusedStream;
        unusedStream << "'" << shouldBeUsedLabels.front() << "'";
        for (std::vector<std::string>::iterator itLabel = shouldBeUsedLabels.begin() + 1,
              itLabelEnd = shouldBeUsedLabels.end();
            itLabel != itLabelEnd;
            ++itLabel) {
          unusedStream << ",'" << *itLabel << "'";
        }
        LogInfo("path")
          << "The following module labels are not assigned to any path:\n"
          << unusedStream.str()
          << "\n";
      }
    }
    if (!unscheduledLabels.empty()) {
      workerManager_.setOnDemandProducts(preg, unscheduledLabels);
    }

    */
    std::set<std::string> usedModuleLabels;
    for( auto const worker: allWorkers()) {
      usedModuleLabels.insert(worker->description().moduleLabel());
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string> >("@all_modules"));
    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    reduceParameterSet(proc_pset, tns.getEndPaths(), modulesInConfig, usedModuleLabels,
                       outputModulePathPositions);
    processEDAliases(proc_pset, processConfiguration->processName(), preg);
    proc_pset.registerIt();
    pset::Registry::instance()->extraForUpdate().setID(proc_pset.id());
    processConfiguration->setParameterSetID(proc_pset.id());
    processConfiguration->setProcessConfigurationID();

    //initializeEarlyDelete(opts,preg,subProcPSet);
    
    // This is used for a little sanity-check to make sure no code
    // modifications alter the number of workers at a later date.
    size_t all_workers_count = allWorkers().size();

    for (auto w : allWorkers()) {

      // All the workers should be in all_workers_ by this point. Thus
      // we can now fill all_output_communicators_.
      auto comm = w->createOutputModuleCommunicator();
      if (comm) {
        all_output_communicators_.emplace_back(boost::shared_ptr<OutputModuleCommunicator>{comm.release()});
      }
    }
    // Now that the output workers are filled in, set any output limits or information.
    limitOutput(proc_pset, branchIDListHelper.branchIDLists());

    loadMissingDictionaries();

    // Sanity check: make sure nobody has added a worker after we've
    // already relied on the WorkerManager being full.
    assert (all_workers_count == allWorkers().size());

    branchIDListHelper.updateRegistries(preg);

    preg.setFrozen();

    for (auto c : all_output_communicators_) {
      c->setEventSelectionInfo(outputModulePathPositions, preg.anyProductProduced());
      c->selectProducts(preg);
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
    for_all(all_output_communicators_, boost::bind(&OutputModuleCommunicator::closeFile, _1));
  }

  void Schedule::openNewOutputFilesIfNeeded() {
    for_all(all_output_communicators_, boost::bind(&OutputModuleCommunicator::openNewFileIfNeeded, _1));
  }

  void Schedule::openOutputFiles(FileBlock& fb) {
    for_all(all_output_communicators_, boost::bind(&OutputModuleCommunicator::openFile, _1, boost::cref(fb)));
  }

  void Schedule::writeRun(RunPrincipal const& rp, ProcessContext const* processContext) {
    for_all(all_output_communicators_, boost::bind(&OutputModuleCommunicator::writeRun, _1, boost::cref(rp), processContext));
  }

  void Schedule::writeLumi(LuminosityBlockPrincipal const& lbp, ProcessContext const* processContext) {
    for_all(all_output_communicators_, boost::bind(&OutputModuleCommunicator::writeLumi, _1, boost::cref(lbp), processContext));
  }

  bool Schedule::shouldWeCloseOutput() const {
    // Return true iff at least one output module returns true.
    return (std::find_if (all_output_communicators_.begin(), all_output_communicators_.end(),
                     boost::bind(&OutputModuleCommunicator::shouldWeCloseFile, _1))
                     != all_output_communicators_.end());
  }

  void Schedule::respondToOpenInputFile(FileBlock const& fb) {
    for_all(allWorkers(), boost::bind(&Worker::respondToOpenInputFile, _1, boost::cref(fb)));
  }

  void Schedule::respondToCloseInputFile(FileBlock const& fb) {
    for_all(allWorkers(), boost::bind(&Worker::respondToCloseInputFile, _1, boost::cref(fb)));
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

  void Schedule::preForkReleaseResources() {
    for_all(allWorkers(), boost::bind(&Worker::preForkReleaseResources, _1));
  }
  void Schedule::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    for_all(allWorkers(), boost::bind(&Worker::postForkReacquireResources, _1, iChildIndex, iNumberOfChildren));
  }

  bool Schedule::changeModule(std::string const& iLabel,
                              ParameterSet const& iPSet) {
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
    
    auto newMod = replaceModule(moduleRegistry_,iLabel,iPSet);

    globalSchedule_->replaceModule(newMod,iLabel);

    for(auto s: streamSchedules_) {
      s->replaceModule(newMod,iLabel);
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
    for(auto& s: streamSchedules_) {
      s->getTriggerTimingReport(rep);
    }
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
}
