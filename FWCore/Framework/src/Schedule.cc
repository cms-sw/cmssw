#include "FWCore/Framework/interface/Schedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
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
      TO temp(from.size());
      transform_into(from.begin(), from.end(), temp.begin(), func);
      to.swap(temp);
    }

    // -----------------------------

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    Schedule::WorkerPtr
    makeInserter(ParameterSet& proc_pset,
                 ProductRegistry& preg,
                 ExceptionToActionTable const& actions,
                 boost::shared_ptr<ActivityRegistry> areg,
                 boost::shared_ptr<ProcessConfiguration> processConfiguration,
                 Schedule::TrigResPtr trptr) {

      ParameterSet* trig_pset = proc_pset.getPSetForUpdate("@trigger_paths");
      trig_pset->registerIt();

      WorkerParams work_args(trig_pset, preg, processConfiguration, actions);
      ModuleDescription md(trig_pset->id(),
                           "TriggerResultInserter",
                           "TriggerResults",
                           processConfiguration.get(),
                           ModuleDescription::getUniqueID());

      areg->preModuleConstructionSignal_(md);
      std::unique_ptr<EDProducer> producer(new TriggerResultInserter(*trig_pset, trptr));
      areg->postModuleConstructionSignal_(md);

      Schedule::WorkerPtr ptr(new WorkerT<EDProducer>(std::move(producer), md, work_args));
      ptr->setActivityRegistry(areg);
      return ptr;
    }

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
                     StreamID streamID,
                     ProcessContext const* processContext) :
    workerManager_(areg, actions),
    actReg_(areg),
    state_(Ready),
    trig_name_list_(tns.getTrigPaths()),
    end_path_name_list_(tns.getEndPaths()),
    results_(new HLTGlobalStatus(trig_name_list_.size())),
    endpath_results_(), // delay!
    results_inserter_(),
    all_output_communicators_(),
    trig_paths_(),
    end_paths_(),
    wantSummary_(tns.wantSummary()),
    total_events_(),
    total_passed_(),
    stopwatch_(wantSummary_? new RunStopwatch::StopwatchPointer::element_type : static_cast<RunStopwatch::StopwatchPointer::element_type*> (nullptr)),
    streamID_(streamID),
    streamContext_(streamID_, processContext),
    endpathsAreActive_(true) {

    ParameterSet const& opts = proc_pset.getUntrackedParameterSet("options", ParameterSet());
    bool hasPath = false;

    int trig_bitpos = 0;
    vstring labelsOnTriggerPaths;
    for (vstring::const_iterator i = trig_name_list_.begin(),
           e = trig_name_list_.end();
         i != e;
         ++i) {
      fillTrigPath(proc_pset, preg, processConfiguration, trig_bitpos, *i, results_, &labelsOnTriggerPaths);
      ++trig_bitpos;
      hasPath = true;
    }

    if (hasPath) {
      // the results inserter stands alone
      results_inserter_ = makeInserter(proc_pset,
                                       preg,
                                       actions, actReg_, processConfiguration, results_);
      addToAllWorkers(results_inserter_.get());
    }

    TrigResPtr epptr(new HLTGlobalStatus(end_path_name_list_.size()));
    endpath_results_ = epptr;

    // fill normal endpaths
    vstring::iterator eib(end_path_name_list_.begin()), eie(end_path_name_list_.end());
    for (int bitpos = 0; eib != eie; ++eib, ++bitpos) {
      fillEndPath(proc_pset, preg, processConfiguration, bitpos, *eib);
    }

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

    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    reduceParameterSet(proc_pset, modulesInConfig, modulesInConfigSet, labelsOnTriggerPaths, shouldBeUsedLabels, outputModulePathPositions);

    processEDAliases(proc_pset, processConfiguration->processName(), preg);

    proc_pset.registerIt();
    pset::Registry::instance()->extraForUpdate().setID(proc_pset.id());
    processConfiguration->setParameterSetID(proc_pset.id());
    processConfiguration->setProcessConfigurationID();

    initializeEarlyDelete(opts,preg,subProcPSet);
    
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

  
  void Schedule::initializeEarlyDelete(edm::ParameterSet const& opts, edm::ProductRegistry const& preg, 
                                       edm::ParameterSet const* subProcPSet) {
    //for now, if have a subProcess, don't allow early delete
    // In the future we should use the SubProcess's 'keep list' to decide what can be kept
    if(subProcPSet)  return;

    //see if 'canDeleteEarly' was set and if so setup the list with those products actually
    // registered for this job
    std::multimap<std::string,Worker*> branchToReadingWorker;
    initializeBranchToReadingWorker(opts,preg,branchToReadingWorker);
    
    //If no delete early items have been specified we don't have to do anything
    if(branchToReadingWorker.size()==0) {
      return;
    }
    const std::vector<std::string> kEmpty;
    std::map<Worker*,unsigned int> reserveSizeForWorker;
    unsigned int upperLimitOnReadingWorker =0;
    unsigned int upperLimitOnIndicies = 0;
    unsigned int nUniqueBranchesToDelete=branchToReadingWorker.size();
    for (auto w :allWorkers()) {
      auto comm = w->createOutputModuleCommunicator();
      if (comm) {
        if(branchToReadingWorker.size()>0) {
          //If an OutputModule needs a product, we can't delete it early
          // so we should remove it from our list
          SelectionsArray const&kept = comm->keptProducts();
          for( auto const& item: kept[InEvent]) {
            auto found = branchToReadingWorker.equal_range(item->branchName());
            if(found.first !=found.second) {
              --nUniqueBranchesToDelete;
              branchToReadingWorker.erase(found.first,found.second);
            }
          }
        }
      } else {
        if(branchToReadingWorker.size()>0) {
          //determine if this module could read a branch we want to delete early
          auto pset = pset::Registry::instance()->getMapped(w->description().parameterSetID());
          if(0!=pset) {
            auto branches = pset->getUntrackedParameter<std::vector<std::string>>("mightGet",kEmpty);
            if(not branches.empty()) {
              ++upperLimitOnReadingWorker;
            }
            for(auto const& branch:branches){ 
              auto found = branchToReadingWorker.equal_range(branch);
              if(found.first != found.second) {
                ++upperLimitOnIndicies;
                ++reserveSizeForWorker[w];
                if(nullptr == found.first->second) {
                  found.first->second = w;
                } else {
                  branchToReadingWorker.insert(make_pair(found.first->first,w));
                }
              }
            }
          }
        }
      }
    }
    {
      auto it = branchToReadingWorker.begin();
      std::vector<std::string> unusedBranches;
      while(it !=branchToReadingWorker.end()) {
        if(it->second == nullptr) {
          unusedBranches.push_back(it->first);
          //erasing the object invalidates the iterator so must advance it first
          auto temp = it;
          ++it;
          branchToReadingWorker.erase(temp);
        } else {
          ++it;
        }
      }
      if(not unusedBranches.empty()) {
        LogWarning l("UnusedProductsForCanDeleteEarly");
        l<<"The following products in the 'canDeleteEarly' list are not used in this job and will be ignored.\n"
        " If possible, remove the producer from the job or add the product to the producer's own 'mightGet' list.";
        for(auto const& n:unusedBranches){
          l<<"\n "<<n;
        }
      }
    }  
    if(0!=branchToReadingWorker.size()) {
      earlyDeleteHelpers_.reserve(upperLimitOnReadingWorker);
      earlyDeleteHelperToBranchIndicies_.resize(upperLimitOnIndicies,0);
      earlyDeleteBranchToCount_.reserve(nUniqueBranchesToDelete);
      std::map<const Worker*,EarlyDeleteHelper*> alreadySeenWorkers;
      std::string lastBranchName;
      size_t nextOpenIndex = 0;
      unsigned int* beginAddress = &(earlyDeleteHelperToBranchIndicies_.front());
      for(auto& branchAndWorker:branchToReadingWorker) {
        if(lastBranchName != branchAndWorker.first) {
          //have to put back the period we removed earlier in order to get the proper name
          BranchID bid(branchAndWorker.first+".");
          earlyDeleteBranchToCount_.emplace_back(std::make_pair(bid,0U));
          lastBranchName = branchAndWorker.first;
        }
        auto found = alreadySeenWorkers.find(branchAndWorker.second);
        if(alreadySeenWorkers.end() == found) {
          //NOTE: we will set aside enough space in earlyDeleteHelperToBranchIndicies_ to accommodate
          // all the branches that might be read by this worker. However, initially we will only tell the
          // EarlyDeleteHelper about the first one. As additional branches are added via 'appendIndex' the
          // EarlyDeleteHelper will automatically advance its internal end pointer.
          size_t index = nextOpenIndex;
          size_t nIndices = reserveSizeForWorker[branchAndWorker.second];
          earlyDeleteHelperToBranchIndicies_[index]=earlyDeleteBranchToCount_.size()-1;
          earlyDeleteHelpers_.emplace_back(EarlyDeleteHelper(beginAddress+index,
                                                             beginAddress+index+1,
                                                             &earlyDeleteBranchToCount_));
          branchAndWorker.second->setEarlyDeleteHelper(&(earlyDeleteHelpers_.back()));
          alreadySeenWorkers.insert(std::make_pair(branchAndWorker.second,&(earlyDeleteHelpers_.back())));
          nextOpenIndex +=nIndices;
        } else {
          found->second->appendIndex(earlyDeleteBranchToCount_.size()-1);
        }
      }
      
      //Now we can compactify the earlyDeleteHelperToBranchIndicies_ since we may have over estimated the
      // space needed for each module
      auto itLast = earlyDeleteHelpers_.begin();
      for(auto it = earlyDeleteHelpers_.begin()+1;it != earlyDeleteHelpers_.end();++it) {
        if(itLast->end() != it->begin()) {
          //figure the offset for next Worker since it hasn't been moved yet so it has the original address
          unsigned int delta = it->begin()- itLast->end();
          it->shiftIndexPointers(delta);
          
          earlyDeleteHelperToBranchIndicies_.erase(earlyDeleteHelperToBranchIndicies_.begin()+
                                                   (itLast->end()-beginAddress),
                                                   earlyDeleteHelperToBranchIndicies_.begin()+
                                                   (it->begin()-beginAddress));
        }
        itLast = it;
      }
      earlyDeleteHelperToBranchIndicies_.erase(earlyDeleteHelperToBranchIndicies_.begin()+(itLast->end()-beginAddress),
                                               earlyDeleteHelperToBranchIndicies_.end());
      
      //now tell the paths about the deleters
      for(auto& p : trig_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      for(auto& p : end_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      resetEarlyDelete();
    }
  }

  void Schedule::reduceParameterSet(ParameterSet& proc_pset,
                                    vstring& modulesInConfig,
                                    std::set<std::string> const& modulesInConfigSet,
                                    vstring& labelsOnTriggerPaths,
                                    vstring& shouldBeUsedLabels,
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
    vstring labelsToBeDropped;
    vstring outputModuleLabels;
    std::string edmType;
    std::string const moduleEdmType("@module_edm_type");
    std::string const outputModule("OutputModule");
    std::string const edAnalyzer("EDAnalyzer");
    std::string const edFilter("EDFilter");
    std::string const edProducer("EDProducer");
    sort_all(labelsOnTriggerPaths);
    vstring::const_iterator iLabelsOnTriggerPaths = labelsOnTriggerPaths.begin();
    vstring::const_iterator endLabelsOnTriggerPaths = labelsOnTriggerPaths.end();
    sort_all(shouldBeUsedLabels);
    vstring::const_iterator iShouldBeUsedLabels = shouldBeUsedLabels.begin();
    vstring::const_iterator endShouldBeUsedLabels = shouldBeUsedLabels.end();

    for (std::set<std::string>::const_iterator i = modulesInConfigSet.begin(),
	   e = modulesInConfigSet.end(); i != e; ++i) {
      edmType = proc_pset.getParameterSet(*i).getParameter<std::string>(moduleEdmType);
      if (edmType == outputModule) {
        labelsToBeDropped.push_back(*i);
        outputModuleLabels.push_back(*i);
      }
      else if (edmType == edAnalyzer) {
        while (iLabelsOnTriggerPaths != endLabelsOnTriggerPaths &&
               *iLabelsOnTriggerPaths < *i) {
          ++iLabelsOnTriggerPaths;
        }
        if (iLabelsOnTriggerPaths == endLabelsOnTriggerPaths ||
            *iLabelsOnTriggerPaths != *i) {
          labelsToBeDropped.push_back(*i);
        }
      }
      else if (edmType == edFilter || edmType == edProducer) {
        while (iShouldBeUsedLabels != endShouldBeUsedLabels &&
               *iShouldBeUsedLabels < *i) {
          ++iShouldBeUsedLabels;
        }
        if (iShouldBeUsedLabels != endShouldBeUsedLabels &&
            *iShouldBeUsedLabels == *i) {
          labelsToBeDropped.push_back(*i);
        }
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
    for (vstring::iterator iEndPath = end_path_name_list_.begin(), endEndPath = end_path_name_list_.end();
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

  void Schedule::fillWorkers(ParameterSet& proc_pset,
                             ProductRegistry& preg,
                             boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                             std::string const& name,
                             bool ignoreFilters,
                             PathWorkers& out,
                             vstring* labelsOnPaths) {
    vstring modnames = proc_pset.getParameter<vstring>(name);
    PathWorkers tmpworkers;

    for (auto const& name : modnames) {

      if (labelsOnPaths) labelsOnPaths->push_back(name);

      WorkerInPath::FilterAction filterAction = WorkerInPath::Normal;
      if (name[0] == '!')       filterAction = WorkerInPath::Veto;
      else if (name[0] == '-')  filterAction = WorkerInPath::Ignore;

      std::string moduleLabel = name;
      if (filterAction != WorkerInPath::Normal) moduleLabel.erase(0, 1);

      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset == 0) {
        std::string pathType("endpath");
        if (!search_all(end_path_name_list_, name)) {
          pathType = std::string("path");
        }
        throw Exception(errors::Configuration) <<
          "The unknown module label \"" << moduleLabel <<
          "\" appears in " << pathType << " \"" << name <<
          "\"\n please check spelling or remove that label from the path.";
      }
      assert(isTracked);

      Worker* worker = workerManager_.getWorker(*modpset, preg, processConfiguration, moduleLabel);
      if (ignoreFilters && filterAction != WorkerInPath::Ignore && worker->moduleType()==Worker::kFilter) {
        // We have a filter on an end path, and the filter is not explicitly ignored.
        // See if the filter is allowed.
        std::vector<std::string> allowed_filters = proc_pset.getUntrackedParameter<vstring>("@filters_on_endpaths");
        if (!search_all(allowed_filters, worker->description().moduleName())) {
          // Filter is not allowed. Ignore the result, and issue a warning.
          filterAction = WorkerInPath::Ignore;
          LogWarning("FilterOnEndPath")
            << "The EDFilter '" << worker->description().moduleName() << "' with module label '" << moduleLabel << "' appears on EndPath '" << name << "'.\n"
            << "The return value of the filter will be ignored.\n"
            << "To suppress this warning, either remove the filter from the endpath,\n"
            << "or explicitly ignore it in the configuration by using cms.ignore().\n";
        }
      }
      tmpworkers.emplace_back(worker, filterAction);
    }

    out.swap(tmpworkers);
  }

  void Schedule::fillTrigPath(ParameterSet& proc_pset,
                              ProductRegistry& preg,
                              boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                              int bitpos, std::string const& name, TrigResPtr trptr,
                              vstring* labelsOnTriggerPaths) {
    PathWorkers tmpworkers;
    Workers holder;
    fillWorkers(proc_pset, preg, processConfiguration, name, false, tmpworkers, labelsOnTriggerPaths);

    for (PathWorkers::iterator wi(tmpworkers.begin()),
          we(tmpworkers.end()); wi != we; ++wi) {
      holder.push_back(wi->getWorker());
    }

    // an empty path will cause an extra bit that is not used
    if (!tmpworkers.empty()) {
      Path p(bitpos, name, tmpworkers, trptr, actionTable(), actReg_, false, &streamContext_);
      if (wantSummary_) {
        p.useStopwatch();
      }
      trig_paths_.push_back(p);
    } else {
      empty_trig_paths_.push_back(bitpos);
      empty_trig_path_names_.push_back(name);
    }
    for_all(holder, boost::bind(&Schedule::addToAllWorkers, this, _1));
  }

  void Schedule::fillEndPath(ParameterSet& proc_pset,
                             ProductRegistry& preg,
                             boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                             int bitpos, std::string const& name) {
    PathWorkers tmpworkers;
    fillWorkers(proc_pset, preg, processConfiguration, name, true, tmpworkers, 0);
    Workers holder;

    for (PathWorkers::iterator wi(tmpworkers.begin()), we(tmpworkers.end()); wi != we; ++wi) {
      holder.push_back(wi->getWorker());
    }

    if (!tmpworkers.empty()) {
      Path p(bitpos, name, tmpworkers, endpath_results_, actionTable(), actReg_, true, &streamContext_);
      if (wantSummary_) {
        p.useStopwatch();
      }
      end_paths_.push_back(p);
    }
    for_all(holder, boost::bind(&Schedule::addToAllWorkers, this, _1));
  }

  void Schedule::endJob(ExceptionCollector & collector) {
    workerManager_.endJob(collector);
    if (collector.hasThrown()) {
      return;
    }

    if (wantSummary_ == false) return;

    TrigPaths::const_iterator pi, pe;

    // The trigger report (pass/fail etc.):

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "---------- Event  Summary ------------";
    if(!trig_paths_.empty()) {
      LogVerbatim("FwkSummary") << "TrigReport"
                                << " Events total = " << totalEvents()
                                << " passed = " << totalEventsPassed()
                                << " failed = " << (totalEventsFailed())
                                << "";
    } else {
      LogVerbatim("FwkSummary") << "TrigReport"
                                << " Events total = " << totalEvents()
                                << " passed = " << totalEvents()
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
    pi = trig_paths_.begin();
    pe = trig_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TrigReport "
                                << std::right << std::setw(5) << 1
                                << std::right << std::setw(5) << pi->bitPosition() << " "
                                << std::right << std::setw(10) << pi->timesRun() << " "
                                << std::right << std::setw(10) << pi->timesPassed() << " "
                                << std::right << std::setw(10) << pi->timesFailed() << " "
                                << std::right << std::setw(10) << pi->timesExcept() << " "
                                << pi->name() << "";
    }

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

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "-------End-Path   Summary ------------";
    LogVerbatim("FwkSummary") << "TrigReport "
                              << std::right << std::setw(10) << "Trig Bit#" << " "
                              << std::right << std::setw(10) << "Run" << " "
                              << std::right << std::setw(10) << "Passed" << " "
                              << std::right << std::setw(10) << "Failed" << " "
                              << std::right << std::setw(10) << "Error" << " "
                              << "Name" << "";
    pi = end_paths_.begin();
    pe = end_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TrigReport "
                                << std::right << std::setw(5) << 0
                                << std::right << std::setw(5) << pi->bitPosition() << " "
                                << std::right << std::setw(10) << pi->timesRun() << " "
                                << std::right << std::setw(10) << pi->timesPassed() << " "
                                << std::right << std::setw(10) << pi->timesFailed() << " "
                                << std::right << std::setw(10) << pi->timesExcept() << " "
                                << pi->name() << "";
    }

    pi = trig_paths_.begin();
    pe = trig_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "---------- Modules in Path: " << pi->name() << " ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
                                << std::right << std::setw(10) << "Trig Bit#" << " "
                                << std::right << std::setw(10) << "Visited" << " "
                                << std::right << std::setw(10) << "Passed" << " "
                                << std::right << std::setw(10) << "Failed" << " "
                                << std::right << std::setw(10) << "Error" << " "
                                << "Name" << "";

      for (unsigned int i = 0; i < pi->size(); ++i) {
        LogVerbatim("FwkSummary") << "TrigReport "
                                  << std::right << std::setw(5) << 1
                                  << std::right << std::setw(5) << pi->bitPosition() << " "
                                  << std::right << std::setw(10) << pi->timesVisited(i) << " "
                                  << std::right << std::setw(10) << pi->timesPassed(i) << " "
                                  << std::right << std::setw(10) << pi->timesFailed(i) << " "
                                  << std::right << std::setw(10) << pi->timesExcept(i) << " "
                                  << pi->getWorker(i)->description().moduleLabel() << "";
      }
    }

    pi = end_paths_.begin();
    pe = end_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "------ Modules in End-Path: " << pi->name() << " ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
                                << std::right << std::setw(10) << "Trig Bit#" << " "
                                << std::right << std::setw(10) << "Visited" << " "
                                << std::right << std::setw(10) << "Passed" << " "
                                << std::right << std::setw(10) << "Failed" << " "
                                << std::right << std::setw(10) << "Error" << " "
                                << "Name" << "";

      for (unsigned int i = 0; i < pi->size(); ++i) {
        LogVerbatim("FwkSummary") << "TrigReport "
                                  << std::right << std::setw(5) << 0
                                  << std::right << std::setw(5) << pi->bitPosition() << " "
                                  << std::right << std::setw(10) << pi->timesVisited(i) << " "
                                  << std::right << std::setw(10) << pi->timesPassed(i) << " "
                                  << std::right << std::setw(10) << pi->timesFailed(i) << " "
                                  << std::right << std::setw(10) << pi->timesExcept(i) << " "
                                  << pi->getWorker(i)->description().moduleLabel() << "";
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
    for (auto const& worker : allWorkers()) {
      LogVerbatim("FwkSummary") << "TrigReport "
                                << std::right << std::setw(10) << worker->timesVisited() << " "
                                << std::right << std::setw(10) << worker->timesRun() << " "
                                << std::right << std::setw(10) << worker->timesPassed() << " "
                                << std::right << std::setw(10) << worker->timesFailed() << " "
                                << std::right << std::setw(10) << worker->timesExcept() << " "
                                << worker->description().moduleLabel() << "";

    }
    LogVerbatim("FwkSummary") << "";

    // The timing report (CPU and Real Time):

    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Event  Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport"
                              << std::setprecision(6) << std::fixed
                              << " CPU/event = " << timeCpuReal().first/std::max(1, totalEvents())
                              << " Real/event = " << timeCpuReal().second/std::max(1, totalEvents())
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
    pi = trig_paths_.begin();
    pe = trig_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1, pi->timesRun()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1, pi->timesRun()) << " "
                                << pi->name() << "";
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
    pi = end_paths_.begin();
    pe = end_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1, pi->timesRun()) << " "
                                << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1, pi->timesRun()) << " "
                                << pi->name() << "";
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

    pi = trig_paths_.begin();
    pe = trig_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "---------- Modules in Path: " << pi->name() << " ---[sec]----";
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
      for (unsigned int i = 0; i < pi->size(); ++i) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1, totalEvents()) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1, totalEvents()) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1, pi->timesVisited(i)) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1, pi->timesVisited(i)) << " "
                                  << pi->getWorker(i)->description().moduleLabel() << "";
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

    pi = end_paths_.begin();
    pe = end_paths_.end();
    for (; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "------ Modules in End-Path: " << pi->name() << " ---[sec]----";
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
      for (unsigned int i = 0; i < pi->size(); ++i) {
        LogVerbatim("FwkSummary") << "TimeReport "
                                  << std::setprecision(6) << std::fixed
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1, totalEvents()) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1, totalEvents()) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1, pi->timesVisited(i)) << " "
                                  << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1, pi->timesVisited(i)) << " "
                                  << pi->getWorker(i)->description().moduleLabel() << "";
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
    for (auto const& worker : allWorkers()) {
      LogVerbatim("FwkSummary") << "TimeReport "
                                << std::setprecision(6) << std::fixed
                                << std::right << std::setw(10) << worker->timeCpuReal().first/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << worker->timeCpuReal().second/std::max(1, totalEvents()) << " "
                                << std::right << std::setw(10) << worker->timeCpuReal().first/std::max(1, worker->timesRun()) << " "
                                << std::right << std::setw(10) << worker->timeCpuReal().second/std::max(1, worker->timesRun()) << " "
                                << std::right << std::setw(10) << worker->timeCpuReal().first/std::max(1, worker->timesVisited()) << " "
                                << std::right << std::setw(10) << worker->timeCpuReal().second/std::max(1, worker->timesVisited()) << " "
                                << worker->description().moduleLabel() << "";
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
    workerManager_.beginJob(iRegistry);
  }
  
  void Schedule::beginStream() {
    workerManager_.beginStream(streamID_, streamContext_);
  }
  
  void Schedule::endStream() {
    workerManager_.endStream(streamID_, streamContext_);
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

    std::auto_ptr<Maker> wm(MakerPluginFactory::get()->create(found->description().moduleName()));
    wm->swapModule(found, iPSet);
    found->beginJob();
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

  void
  Schedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    oLabelsToFill.reserve(trig_paths_.size());
    std::transform(trig_paths_.begin(),
                   trig_paths_.end(),
                   std::back_inserter(oLabelsToFill),
                   boost::bind(&Path::name, _1));
  }

  void
  Schedule::modulesInPath(std::string const& iPathLabel,
                          std::vector<std::string>& oLabelsToFill) const {
    TrigPaths::const_iterator itFound =
    std::find_if (trig_paths_.begin(),
                 trig_paths_.end(),
                 boost::bind(std::equal_to<std::string>(),
                             iPathLabel,
                             boost::bind(&Path::name, _1)));
    if (itFound!=trig_paths_.end()) {
      oLabelsToFill.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        oLabelsToFill.push_back(itFound->getWorker(i)->description().moduleLabel());
      }
    }
  }

  void
  Schedule::enableEndPaths(bool active) {
    endpathsAreActive_ = active;
  }

  bool
  Schedule::endPathsEnabled() const {
    return endpathsAreActive_;
  }

  void
  fillModuleInPathSummary(Path const&, ModuleInPathSummary&) {
  }

  void
  fillModuleInPathSummary(Path const& path,
                          size_t which,
                          ModuleInPathSummary& sum) {
    sum.timesVisited = path.timesVisited(which);
    sum.timesPassed  = path.timesPassed(which);
    sum.timesFailed  = path.timesFailed(which);
    sum.timesExcept  = path.timesExcept(which);
    sum.moduleLabel  = path.getWorker(which)->description().moduleLabel();
  }

  void
  fillPathSummary(Path const& path, PathSummary& sum) {
    sum.name        = path.name();
    sum.bitPosition = path.bitPosition();
    sum.timesRun    = path.timesRun();
    sum.timesPassed = path.timesPassed();
    sum.timesFailed = path.timesFailed();
    sum.timesExcept = path.timesExcept();

    Path::size_type sz = path.size();
    std::vector<ModuleInPathSummary> temp(sz);
    for (size_t i = 0; i != sz; ++i) {
      fillModuleInPathSummary(path, i, temp[i]);
    }
    sum.moduleInPathSummaries.swap(temp);
  }

  void
  fillWorkerSummaryAux(Worker const& w, WorkerSummary& sum) {
    sum.timesVisited = w.timesVisited();
    sum.timesRun     = w.timesRun();
    sum.timesPassed  = w.timesPassed();
    sum.timesFailed  = w.timesFailed();
    sum.timesExcept  = w.timesExcept();
    sum.moduleLabel  = w.description().moduleLabel();
  }

  void
  fillWorkerSummary(Worker const* pw, WorkerSummary& sum) {
    fillWorkerSummaryAux(*pw, sum);
  }

  void
  Schedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents = totalEvents();
    rep.eventSummary.totalEventsPassed = totalEventsPassed();
    rep.eventSummary.totalEventsFailed = totalEventsFailed();

    fill_summary(trig_paths_,  rep.trigPathSummaries, &fillPathSummary);
    fill_summary(end_paths_,   rep.endPathSummaries,  &fillPathSummary);
    fill_summary(allWorkers(), rep.workerSummaries,   &fillWorkerSummary);
  }

  void
  Schedule::clearCounters() {
    total_events_ = total_passed_ = 0;
    for_all(trig_paths_, boost::bind(&Path::clearCounters, _1));
    for_all(end_paths_, boost::bind(&Path::clearCounters, _1));
    for_all(allWorkers(), boost::bind(&Worker::clearCounters, _1));
  }

  void
  Schedule::resetAll() {
    results_->reset();
    endpath_results_->reset();
  }

  void
  Schedule::addToAllWorkers(Worker* w) {
    workerManager_.addToAllWorkers(w, wantSummary_);
  }

  void 
  Schedule::resetEarlyDelete() {
    //must be sure we have cleared the count first
    for(auto& count:earlyDeleteBranchToCount_) {
      count.second = 0;
    }
    //now reset based on how many helpers use that branch
    for(auto& index: earlyDeleteHelperToBranchIndicies_) {
      ++(earlyDeleteBranchToCount_[index].second);
    }
    for(auto& helper: earlyDeleteHelpers_) {
      helper.reset();
    }
  }

}
