#include "FWCore/Framework/src/StreamSchedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/interface/TriggerTimingReport.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/ModuleRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "LuminosityBlockProcessingStatus.h"

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
    void transform_into(InputIterator begin, InputIterator end, ForwardIterator out, Func func) {
      for (; begin != end; ++begin, ++out)
        func(*begin, *out);
    }

    // Function template that takes a sequence 'from', a sequence
    // 'to', and a callable object 'func'. It and applies
    // transform_into to fill the 'to' sequence with the values
    // calcuated by the callable object, taking care to fill the
    // outupt only if all calls succeed.
    template <typename FROM, typename TO, typename FUNC>
    void fill_summary(FROM const& from, TO& to, FUNC func) {
      if (to.size() != from.size()) {
        TO temp(from.size());
        transform_into(from.begin(), from.end(), temp.begin(), func);
        to.swap(temp);
      } else {
        transform_into(from.begin(), from.end(), to.begin(), func);
      }
    }

    // -----------------------------

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    StreamSchedule::WorkerPtr makeInserter(ExceptionToActionTable const& actions,
                                           std::shared_ptr<ActivityRegistry> areg,
                                           std::shared_ptr<TriggerResultInserter> inserter) {
      StreamSchedule::WorkerPtr ptr(
          new edm::WorkerT<TriggerResultInserter::ModuleType>(inserter, inserter->moduleDescription(), &actions));
      ptr->setActivityRegistry(areg);
      return ptr;
    }

    void initializeBranchToReadingWorker(ParameterSet const& opts,
                                         ProductRegistry const& preg,
                                         std::multimap<std::string, Worker*>& branchToReadingWorker) {
      // See if any data has been marked to be deleted early (removing any duplicates)
      auto vBranchesToDeleteEarly = opts.getUntrackedParameter<std::vector<std::string>>("canDeleteEarly");
      if (not vBranchesToDeleteEarly.empty()) {
        std::sort(vBranchesToDeleteEarly.begin(), vBranchesToDeleteEarly.end(), std::less<std::string>());
        vBranchesToDeleteEarly.erase(std::unique(vBranchesToDeleteEarly.begin(), vBranchesToDeleteEarly.end()),
                                     vBranchesToDeleteEarly.end());

        // Are the requested items in the product registry?
        auto allBranchNames = preg.allBranchNames();
        //the branch names all end with a period, which we do not want to compare with
        for (auto& b : allBranchNames) {
          b.resize(b.size() - 1);
        }
        std::sort(allBranchNames.begin(), allBranchNames.end(), std::less<std::string>());
        std::vector<std::string> temp;
        temp.reserve(vBranchesToDeleteEarly.size());

        std::set_intersection(vBranchesToDeleteEarly.begin(),
                              vBranchesToDeleteEarly.end(),
                              allBranchNames.begin(),
                              allBranchNames.end(),
                              std::back_inserter(temp));
        vBranchesToDeleteEarly.swap(temp);
        if (temp.size() != vBranchesToDeleteEarly.size()) {
          std::vector<std::string> missingProducts;
          std::set_difference(temp.begin(),
                              temp.end(),
                              vBranchesToDeleteEarly.begin(),
                              vBranchesToDeleteEarly.end(),
                              std::back_inserter(missingProducts));
          LogInfo l("MissingProductsForCanDeleteEarly");
          l << "The following products in the 'canDeleteEarly' list are not available in this job and will be ignored.";
          for (auto const& n : missingProducts) {
            l << "\n " << n;
          }
        }
        //set placeholder for the branch, we will remove the nullptr if a
        // module actually wants the branch.
        for (auto const& branch : vBranchesToDeleteEarly) {
          branchToReadingWorker.insert(std::make_pair(branch, static_cast<Worker*>(nullptr)));
        }
      }
    }
  }  // namespace

  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  StreamSchedule::StreamSchedule(
      std::shared_ptr<TriggerResultInserter> inserter,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      std::shared_ptr<ModuleRegistry> modReg,
      ParameterSet& proc_pset,
      service::TriggerNamesService const& tns,
      PreallocationConfiguration const& prealloc,
      ProductRegistry& preg,
      BranchIDListHelper& branchIDListHelper,
      ExceptionToActionTable const& actions,
      std::shared_ptr<ActivityRegistry> areg,
      std::shared_ptr<ProcessConfiguration> processConfiguration,
      bool allowEarlyDelete,
      StreamID streamID,
      ProcessContext const* processContext)
      : workerManager_(modReg, areg, actions),
        actReg_(areg),
        results_(new HLTGlobalStatus(tns.getTrigPaths().size())),
        results_inserter_(),
        trig_paths_(),
        end_paths_(),
        total_events_(),
        total_passed_(),
        number_of_unscheduled_modules_(0),
        streamID_(streamID),
        streamContext_(streamID_, processContext),
        endpathsAreActive_(true),
        skippingEvent_(false) {
    ParameterSet const& opts = proc_pset.getUntrackedParameterSet("options", ParameterSet());
    bool hasPath = false;
    std::vector<std::string> const& pathNames = tns.getTrigPaths();
    std::vector<std::string> const& endPathNames = tns.getEndPaths();

    int trig_bitpos = 0;
    trig_paths_.reserve(pathNames.size());
    for (auto const& trig_name : pathNames) {
      fillTrigPath(proc_pset, preg, &prealloc, processConfiguration, trig_bitpos, trig_name, results(), endPathNames);
      ++trig_bitpos;
      hasPath = true;
    }

    if (hasPath) {
      // the results inserter stands alone
      inserter->setTrigResultForStream(streamID.value(), results());

      results_inserter_ = makeInserter(actions, actReg_, inserter);
      addToAllWorkers(results_inserter_.get());
    }

    // fill normal endpaths
    int bitpos = 0;
    end_paths_.reserve(endPathNames.size());
    for (auto const& end_path_name : endPathNames) {
      fillEndPath(proc_pset, preg, &prealloc, processConfiguration, bitpos, end_path_name, endPathNames);
      ++bitpos;
    }

    makePathStatusInserters(pathStatusInserters, endPathStatusInserters, actions);

    //See if all modules were used
    std::set<std::string> usedWorkerLabels;
    for (auto const& worker : allWorkers()) {
      usedWorkerLabels.insert(worker->description().moduleLabel());
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string>>("@all_modules"));
    std::set<std::string> modulesInConfigSet(modulesInConfig.begin(), modulesInConfig.end());
    std::vector<std::string> unusedLabels;
    set_difference(modulesInConfigSet.begin(),
                   modulesInConfigSet.end(),
                   usedWorkerLabels.begin(),
                   usedWorkerLabels.end(),
                   back_inserter(unusedLabels));
    std::set<std::string> unscheduledLabels;
    std::vector<std::string> shouldBeUsedLabels;
    if (!unusedLabels.empty()) {
      //Need to
      // 1) create worker
      // 2) if it is a WorkerT<EDProducer>, add it to our list
      // 3) hand list to our delayed reader
      for (auto const& label : unusedLabels) {
        bool isTracked;
        ParameterSet* modulePSet(proc_pset.getPSetForUpdate(label, isTracked));
        assert(isTracked);
        assert(modulePSet != nullptr);
        workerManager_.addToUnscheduledWorkers(
            *modulePSet, preg, &prealloc, processConfiguration, label, unscheduledLabels, shouldBeUsedLabels);
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
        LogInfo("path") << "The following module labels are not assigned to any path:\n" << unusedStream.str() << "\n";
      }
    }
    number_of_unscheduled_modules_ = unscheduledLabels.size();

    initializeEarlyDelete(*modReg, opts, preg, allowEarlyDelete);

  }  // StreamSchedule::StreamSchedule

  void StreamSchedule::initializeEarlyDelete(ModuleRegistry& modReg,
                                             edm::ParameterSet const& opts,
                                             edm::ProductRegistry const& preg,
                                             bool allowEarlyDelete) {
    //for now, if have a subProcess, don't allow early delete
    // In the future we should use the SubProcess's 'keep list' to decide what can be kept
    if (not allowEarlyDelete)
      return;

    //see if 'canDeleteEarly' was set and if so setup the list with those products actually
    // registered for this job
    std::multimap<std::string, Worker*> branchToReadingWorker;
    initializeBranchToReadingWorker(opts, preg, branchToReadingWorker);

    //If no delete early items have been specified we don't have to do anything
    if (branchToReadingWorker.empty()) {
      return;
    }
    const std::vector<std::string> kEmpty;
    std::map<Worker*, unsigned int> reserveSizeForWorker;
    unsigned int upperLimitOnReadingWorker = 0;
    unsigned int upperLimitOnIndicies = 0;
    unsigned int nUniqueBranchesToDelete = branchToReadingWorker.size();

    //talk with output modules first
    modReg.forAllModuleHolders([&branchToReadingWorker, &nUniqueBranchesToDelete](maker::ModuleHolder* iHolder) {
      auto comm = iHolder->createOutputModuleCommunicator();
      if (comm) {
        if (!branchToReadingWorker.empty()) {
          //If an OutputModule needs a product, we can't delete it early
          // so we should remove it from our list
          SelectedProductsForBranchType const& kept = comm->keptProducts();
          for (auto const& item : kept[InEvent]) {
            BranchDescription const& desc = *item.first;
            auto found = branchToReadingWorker.equal_range(desc.branchName());
            if (found.first != found.second) {
              --nUniqueBranchesToDelete;
              branchToReadingWorker.erase(found.first, found.second);
            }
          }
        }
      }
    });

    if (branchToReadingWorker.empty()) {
      return;
    }

    for (auto w : allWorkers()) {
      //determine if this module could read a branch we want to delete early
      auto pset = pset::Registry::instance()->getMapped(w->description().parameterSetID());
      if (nullptr != pset) {
        auto branches = pset->getUntrackedParameter<std::vector<std::string>>("mightGet", kEmpty);
        if (not branches.empty()) {
          ++upperLimitOnReadingWorker;
        }
        for (auto const& branch : branches) {
          auto found = branchToReadingWorker.equal_range(branch);
          if (found.first != found.second) {
            ++upperLimitOnIndicies;
            ++reserveSizeForWorker[w];
            if (nullptr == found.first->second) {
              found.first->second = w;
            } else {
              branchToReadingWorker.insert(make_pair(found.first->first, w));
            }
          }
        }
      }
    }
    {
      auto it = branchToReadingWorker.begin();
      std::vector<std::string> unusedBranches;
      while (it != branchToReadingWorker.end()) {
        if (it->second == nullptr) {
          unusedBranches.push_back(it->first);
          //erasing the object invalidates the iterator so must advance it first
          auto temp = it;
          ++it;
          branchToReadingWorker.erase(temp);
        } else {
          ++it;
        }
      }
      if (not unusedBranches.empty()) {
        LogWarning l("UnusedProductsForCanDeleteEarly");
        l << "The following products in the 'canDeleteEarly' list are not used in this job and will be ignored.\n"
             " If possible, remove the producer from the job or add the product to the producer's own 'mightGet' list.";
        for (auto const& n : unusedBranches) {
          l << "\n " << n;
        }
      }
    }
    if (!branchToReadingWorker.empty()) {
      earlyDeleteHelpers_.reserve(upperLimitOnReadingWorker);
      earlyDeleteHelperToBranchIndicies_.resize(upperLimitOnIndicies, 0);
      earlyDeleteBranchToCount_.reserve(nUniqueBranchesToDelete);
      std::map<const Worker*, EarlyDeleteHelper*> alreadySeenWorkers;
      std::string lastBranchName;
      size_t nextOpenIndex = 0;
      unsigned int* beginAddress = &(earlyDeleteHelperToBranchIndicies_.front());
      for (auto& branchAndWorker : branchToReadingWorker) {
        if (lastBranchName != branchAndWorker.first) {
          //have to put back the period we removed earlier in order to get the proper name
          BranchID bid(branchAndWorker.first + ".");
          earlyDeleteBranchToCount_.emplace_back(bid, 0U);
          lastBranchName = branchAndWorker.first;
        }
        auto found = alreadySeenWorkers.find(branchAndWorker.second);
        if (alreadySeenWorkers.end() == found) {
          //NOTE: we will set aside enough space in earlyDeleteHelperToBranchIndicies_ to accommodate
          // all the branches that might be read by this worker. However, initially we will only tell the
          // EarlyDeleteHelper about the first one. As additional branches are added via 'appendIndex' the
          // EarlyDeleteHelper will automatically advance its internal end pointer.
          size_t index = nextOpenIndex;
          size_t nIndices = reserveSizeForWorker[branchAndWorker.second];
          earlyDeleteHelperToBranchIndicies_[index] = earlyDeleteBranchToCount_.size() - 1;
          earlyDeleteHelpers_.emplace_back(beginAddress + index, beginAddress + index + 1, &earlyDeleteBranchToCount_);
          branchAndWorker.second->setEarlyDeleteHelper(&(earlyDeleteHelpers_.back()));
          alreadySeenWorkers.insert(std::make_pair(branchAndWorker.second, &(earlyDeleteHelpers_.back())));
          nextOpenIndex += nIndices;
        } else {
          found->second->appendIndex(earlyDeleteBranchToCount_.size() - 1);
        }
      }

      //Now we can compactify the earlyDeleteHelperToBranchIndicies_ since we may have over estimated the
      // space needed for each module
      auto itLast = earlyDeleteHelpers_.begin();
      for (auto it = earlyDeleteHelpers_.begin() + 1; it != earlyDeleteHelpers_.end(); ++it) {
        if (itLast->end() != it->begin()) {
          //figure the offset for next Worker since it hasn't been moved yet so it has the original address
          unsigned int delta = it->begin() - itLast->end();
          it->shiftIndexPointers(delta);

          earlyDeleteHelperToBranchIndicies_.erase(
              earlyDeleteHelperToBranchIndicies_.begin() + (itLast->end() - beginAddress),
              earlyDeleteHelperToBranchIndicies_.begin() + (it->begin() - beginAddress));
        }
        itLast = it;
      }
      earlyDeleteHelperToBranchIndicies_.erase(
          earlyDeleteHelperToBranchIndicies_.begin() + (itLast->end() - beginAddress),
          earlyDeleteHelperToBranchIndicies_.end());

      //now tell the paths about the deleters
      for (auto& p : trig_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      for (auto& p : end_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      resetEarlyDelete();
    }
  }

  void StreamSchedule::fillWorkers(ParameterSet& proc_pset,
                                   ProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   std::string const& pathName,
                                   bool ignoreFilters,
                                   PathWorkers& out,
                                   std::vector<std::string> const& endPathNames) {
    vstring modnames = proc_pset.getParameter<vstring>(pathName);
    PathWorkers tmpworkers;

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

      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset == nullptr) {
        std::string pathType("endpath");
        if (!search_all(endPathNames, pathName)) {
          pathType = std::string("path");
        }
        throw Exception(errors::Configuration)
            << "The unknown module label \"" << moduleLabel << "\" appears in " << pathType << " \"" << pathName
            << "\"\n please check spelling or remove that label from the path.";
      }
      assert(isTracked);

      Worker* worker = workerManager_.getWorker(*modpset, preg, prealloc, processConfiguration, moduleLabel);
      if (ignoreFilters && filterAction != WorkerInPath::Ignore && worker->moduleType() == Worker::kFilter) {
        // We have a filter on an end path, and the filter is not explicitly ignored.
        // See if the filter is allowed.
        std::vector<std::string> allowed_filters = proc_pset.getUntrackedParameter<vstring>("@filters_on_endpaths");
        if (!search_all(allowed_filters, worker->description().moduleName())) {
          // Filter is not allowed. Ignore the result, and issue a warning.
          filterAction = WorkerInPath::Ignore;
          LogWarning("FilterOnEndPath") << "The EDFilter '" << worker->description().moduleName()
                                        << "' with module label '" << moduleLabel << "' appears on EndPath '"
                                        << pathName << "'.\n"
                                        << "The return value of the filter will be ignored.\n"
                                        << "To suppress this warning, either remove the filter from the endpath,\n"
                                        << "or explicitly ignore it in the configuration by using cms.ignore().\n";
        }
      }
      bool runConcurrently = not doNotRunConcurrently;
      if (runConcurrently && worker->moduleType() == Worker::kFilter and filterAction != WorkerInPath::Ignore) {
        runConcurrently = false;
      }
      tmpworkers.emplace_back(worker, filterAction, placeInPath, runConcurrently);
      ++placeInPath;
    }

    out.swap(tmpworkers);
  }

  void StreamSchedule::fillTrigPath(ParameterSet& proc_pset,
                                    ProductRegistry& preg,
                                    PreallocationConfiguration const* prealloc,
                                    std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                    int bitpos,
                                    std::string const& name,
                                    TrigResPtr trptr,
                                    std::vector<std::string> const& endPathNames) {
    PathWorkers tmpworkers;
    fillWorkers(proc_pset, preg, prealloc, processConfiguration, name, false, tmpworkers, endPathNames);

    // an empty path will cause an extra bit that is not used
    if (!tmpworkers.empty()) {
      trig_paths_.emplace_back(bitpos,
                               name,
                               tmpworkers,
                               trptr,
                               actionTable(),
                               actReg_,
                               &streamContext_,
                               &skippingEvent_,
                               PathContext::PathType::kPath);
    } else {
      empty_trig_paths_.push_back(bitpos);
    }
    for (WorkerInPath const& workerInPath : tmpworkers) {
      addToAllWorkers(workerInPath.getWorker());
    }
  }

  void StreamSchedule::fillEndPath(ParameterSet& proc_pset,
                                   ProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   int bitpos,
                                   std::string const& name,
                                   std::vector<std::string> const& endPathNames) {
    PathWorkers tmpworkers;
    fillWorkers(proc_pset, preg, prealloc, processConfiguration, name, true, tmpworkers, endPathNames);

    if (!tmpworkers.empty()) {
      //EndPaths are not supposed to stop if SkipEvent type exception happens
      end_paths_.emplace_back(bitpos,
                              name,
                              tmpworkers,
                              TrigResPtr(),
                              actionTable(),
                              actReg_,
                              &streamContext_,
                              nullptr,
                              PathContext::PathType::kEndPath);
    } else {
      empty_end_paths_.push_back(bitpos);
    }
    for (WorkerInPath const& workerInPath : tmpworkers) {
      addToAllWorkers(workerInPath.getWorker());
    }
  }

  void StreamSchedule::beginStream() { workerManager_.beginStream(streamID_, streamContext_); }

  void StreamSchedule::endStream() { workerManager_.endStream(streamID_, streamContext_); }

  void StreamSchedule::replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel) {
    Worker* found = nullptr;
    for (auto const& worker : allWorkers()) {
      if (worker->description().moduleLabel() == iLabel) {
        found = worker;
        break;
      }
    }
    if (nullptr == found) {
      return;
    }

    iMod->replaceModuleFor(found);
    found->beginStream(streamID_, streamContext_);
  }

  std::vector<ModuleDescription const*> StreamSchedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->descPtr();
      result.push_back(p);
    }
    return result;
  }

  void StreamSchedule::processOneEventAsync(
      WaitingTaskHolder iTask,
      EventPrincipal& ep,
      EventSetupImpl const& es,
      ServiceToken const& serviceToken,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters) {
    // Caught exception is propagated via WaitingTaskHolder
    CMS_SA_ALLOW try {
      this->resetAll();

      using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

      Traits::setStreamContext(streamContext_, ep);
      //a service may want to communicate with another service
      ServiceRegistry::Operate guard(serviceToken);
      Traits::preScheduleSignal(actReg_.get(), &streamContext_);

      HLTPathStatus hltPathStatus(hlt::Pass, 0);
      for (int empty_trig_path : empty_trig_paths_) {
        results_->at(empty_trig_path) = hltPathStatus;
        pathStatusInserters[empty_trig_path]->setPathStatus(streamID_, hltPathStatus);
        std::exception_ptr except = pathStatusInserterWorkers_[empty_trig_path]
                                        ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                                            ep, es, streamID_, ParentContext(&streamContext_), &streamContext_);
        if (except) {
          iTask.doneWaiting(except);
          return;
        }
      }
      for (int empty_end_path : empty_end_paths_) {
        std::exception_ptr except = endPathStatusInserterWorkers_[empty_end_path]
                                        ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                                            ep, es, streamID_, ParentContext(&streamContext_), &streamContext_);
        if (except) {
          iTask.doneWaiting(except);
          return;
        }
      }

      // This call takes care of the unscheduled processing.
      workerManager_.setupOnDemandSystem(ep, es);

      ++total_events_;

      //use to give priorities on an error to ones from Paths
      auto pathErrorHolder = std::make_unique<std::atomic<std::exception_ptr*>>(nullptr);
      auto pathErrorPtr = pathErrorHolder.get();
      auto allPathsDone = make_waiting_task(
          tbb::task::allocate_root(),
          [iTask, this, serviceToken, pathError = std::move(pathErrorHolder)](std::exception_ptr const* iPtr) mutable {
            ServiceRegistry::Operate operate(serviceToken);

            std::exception_ptr ptr;
            if (pathError->load()) {
              ptr = *pathError->load();
              delete pathError->load();
            }
            if ((not ptr) and iPtr) {
              ptr = *iPtr;
            }
            iTask.doneWaiting(finishProcessOneEvent(ptr));
          });
      //The holder guarantees that if the paths finish before the loop ends
      // that we do not start too soon. It also guarantees that the task will
      // run under that condition.
      WaitingTaskHolder allPathsHolder(allPathsDone);

      auto pathsDone = make_waiting_task(
          tbb::task::allocate_root(),
          [allPathsHolder, pathErrorPtr, &ep, &es, this, serviceToken](std::exception_ptr const* iPtr) mutable {
            ServiceRegistry::Operate operate(serviceToken);

            if (iPtr) {
              //this is used to prioritize this error over one
              // that happens in EndPath or Accumulate
              pathErrorPtr->store(new std::exception_ptr(*iPtr));
            }
            finishedPaths(*pathErrorPtr, std::move(allPathsHolder), ep, es);
          });

      //The holder guarantees that if the paths finish before the loop ends
      // that we do not start too soon. It also guarantees that the task will
      // run under that condition.
      WaitingTaskHolder taskHolder(pathsDone);

      //start end paths first so on single threaded the paths will run first
      for (auto it = end_paths_.rbegin(), itEnd = end_paths_.rend(); it != itEnd; ++it) {
        it->processOneOccurrenceAsync(allPathsDone, ep, es, serviceToken, streamID_, &streamContext_);
      }

      for (auto it = trig_paths_.rbegin(), itEnd = trig_paths_.rend(); it != itEnd; ++it) {
        it->processOneOccurrenceAsync(pathsDone, ep, es, serviceToken, streamID_, &streamContext_);
      }

      ParentContext parentContext(&streamContext_);
      workerManager_.processAccumulatorsAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
          allPathsDone, ep, es, serviceToken, streamID_, parentContext, &streamContext_);
    } catch (...) {
      iTask.doneWaiting(std::current_exception());
    }
  }

  void StreamSchedule::finishedPaths(std::atomic<std::exception_ptr*>& iExcept,
                                     WaitingTaskHolder iWait,
                                     EventPrincipal& ep,
                                     EventSetupImpl const& es) {
    if (iExcept) {
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try { std::rethrow_exception(*(iExcept.load())); } catch (cms::Exception& e) {
        exception_actions::ActionCodes action = actionTable().find(e.category());
        assert(action != exception_actions::IgnoreCompletely);
        assert(action != exception_actions::FailPath);
        if (action == exception_actions::SkipEvent) {
          edm::printCmsExceptionWarning("SkipEvent", e);
          *(iExcept.load()) = std::exception_ptr();
        } else {
          *(iExcept.load()) = std::current_exception();
        }
      } catch (...) {
        *(iExcept.load()) = std::current_exception();
      }
    }

    if ((not iExcept) and results_->accept()) {
      ++total_passed_;
    }

    if (nullptr != results_inserter_.get()) {
      // Caught exception is propagated to the caller
      CMS_SA_ALLOW try {
        //Even if there was an exception, we need to allow results inserter
        // to run since some module may be waiting on its results.
        ParentContext parentContext(&streamContext_);
        using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

        results_inserter_->doWork<Traits>(ep, es, streamID_, parentContext, &streamContext_);
      } catch (cms::Exception& ex) {
        if (not iExcept) {
          if (ex.context().empty()) {
            std::ostringstream ost;
            ost << "Processing Event " << ep.id();
            ex.addContext(ost.str());
          }
          iExcept.store(new std::exception_ptr(std::current_exception()));
        }
      } catch (...) {
        if (not iExcept) {
          iExcept.store(new std::exception_ptr(std::current_exception()));
        }
      }
    }
    std::exception_ptr ptr;
    if (iExcept) {
      ptr = *iExcept.load();
    }
    iWait.doneWaiting(ptr);
  }

  std::exception_ptr StreamSchedule::finishProcessOneEvent(std::exception_ptr iExcept) {
    using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

    if (iExcept) {
      //add context information to the exception and print message
      try {
        convertException::wrap([&]() { std::rethrow_exception(iExcept); });
      } catch (cms::Exception& ex) {
        bool const cleaningUpAfterException = false;
        if (ex.context().empty()) {
          addContextAndPrintException("Calling function StreamSchedule::processOneEvent", ex, cleaningUpAfterException);
        } else {
          addContextAndPrintException("", ex, cleaningUpAfterException);
        }
        iExcept = std::current_exception();
      }

      actReg_->preStreamEarlyTerminationSignal_(streamContext_, TerminationOrigin::ExceptionFromThisContext);
    }
    // Caught exception is propagated to the caller
    CMS_SA_ALLOW try { Traits::postScheduleSignal(actReg_.get(), &streamContext_); } catch (...) {
      if (not iExcept) {
        iExcept = std::current_exception();
      }
    }
    if (not iExcept) {
      resetEarlyDelete();
    }

    return iExcept;
  }

  void StreamSchedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    oLabelsToFill.reserve(trig_paths_.size());
    std::transform(trig_paths_.begin(),
                   trig_paths_.end(),
                   std::back_inserter(oLabelsToFill),
                   std::bind(&Path::name, std::placeholders::_1));
  }

  void StreamSchedule::modulesInPath(std::string const& iPathLabel, std::vector<std::string>& oLabelsToFill) const {
    TrigPaths::const_iterator itFound = std::find_if(
        trig_paths_.begin(),
        trig_paths_.end(),
        std::bind(std::equal_to<std::string>(), iPathLabel, std::bind(&Path::name, std::placeholders::_1)));
    if (itFound != trig_paths_.end()) {
      oLabelsToFill.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        oLabelsToFill.push_back(itFound->getWorker(i)->description().moduleLabel());
      }
    }
  }

  void StreamSchedule::moduleDescriptionsInPath(std::string const& iPathLabel,
                                                std::vector<ModuleDescription const*>& descriptions,
                                                unsigned int hint) const {
    descriptions.clear();
    bool found = false;
    TrigPaths::const_iterator itFound;

    if (hint < trig_paths_.size()) {
      itFound = trig_paths_.begin() + hint;
      if (itFound->name() == iPathLabel)
        found = true;
    }
    if (!found) {
      // if the hint did not work, do it the slow way
      itFound = std::find_if(
          trig_paths_.begin(),
          trig_paths_.end(),
          std::bind(std::equal_to<std::string>(), iPathLabel, std::bind(&Path::name, std::placeholders::_1)));
      if (itFound != trig_paths_.end())
        found = true;
    }
    if (found) {
      descriptions.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        descriptions.push_back(itFound->getWorker(i)->descPtr());
      }
    }
  }

  void StreamSchedule::moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                                   std::vector<ModuleDescription const*>& descriptions,
                                                   unsigned int hint) const {
    descriptions.clear();
    bool found = false;
    TrigPaths::const_iterator itFound;

    if (hint < end_paths_.size()) {
      itFound = end_paths_.begin() + hint;
      if (itFound->name() == iEndPathLabel)
        found = true;
    }
    if (!found) {
      // if the hint did not work, do it the slow way
      itFound = std::find_if(
          end_paths_.begin(),
          end_paths_.end(),
          std::bind(std::equal_to<std::string>(), iEndPathLabel, std::bind(&Path::name, std::placeholders::_1)));
      if (itFound != end_paths_.end())
        found = true;
    }
    if (found) {
      descriptions.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        descriptions.push_back(itFound->getWorker(i)->descPtr());
      }
    }
  }

  void StreamSchedule::enableEndPaths(bool active) { endpathsAreActive_ = active; }

  bool StreamSchedule::endPathsEnabled() const { return endpathsAreActive_; }

  static void fillModuleInPathSummary(Path const& path, size_t which, ModuleInPathSummary& sum) {
    sum.timesVisited += path.timesVisited(which);
    sum.timesPassed += path.timesPassed(which);
    sum.timesFailed += path.timesFailed(which);
    sum.timesExcept += path.timesExcept(which);
    sum.moduleLabel = path.getWorker(which)->description().moduleLabel();
  }

  static void fillPathSummary(Path const& path, PathSummary& sum) {
    sum.name = path.name();
    sum.bitPosition = path.bitPosition();
    sum.timesRun += path.timesRun();
    sum.timesPassed += path.timesPassed();
    sum.timesFailed += path.timesFailed();
    sum.timesExcept += path.timesExcept();

    Path::size_type sz = path.size();
    if (sum.moduleInPathSummaries.empty()) {
      std::vector<ModuleInPathSummary> temp(sz);
      for (size_t i = 0; i != sz; ++i) {
        fillModuleInPathSummary(path, i, temp[i]);
      }
      sum.moduleInPathSummaries.swap(temp);
    } else {
      assert(sz == sum.moduleInPathSummaries.size());
      for (size_t i = 0; i != sz; ++i) {
        fillModuleInPathSummary(path, i, sum.moduleInPathSummaries[i]);
      }
    }
  }

  static void fillWorkerSummaryAux(Worker const& w, WorkerSummary& sum) {
    sum.timesVisited += w.timesVisited();
    sum.timesRun += w.timesRun();
    sum.timesPassed += w.timesPassed();
    sum.timesFailed += w.timesFailed();
    sum.timesExcept += w.timesExcept();
    sum.moduleLabel = w.description().moduleLabel();
  }

  static void fillWorkerSummary(Worker const* pw, WorkerSummary& sum) { fillWorkerSummaryAux(*pw, sum); }

  void StreamSchedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents += totalEvents();
    rep.eventSummary.totalEventsPassed += totalEventsPassed();
    rep.eventSummary.totalEventsFailed += totalEventsFailed();

    fill_summary(trig_paths_, rep.trigPathSummaries, &fillPathSummary);
    fill_summary(end_paths_, rep.endPathSummaries, &fillPathSummary);
    fill_summary(allWorkers(), rep.workerSummaries, &fillWorkerSummary);
  }

  void StreamSchedule::clearCounters() {
    using std::placeholders::_1;
    total_events_ = total_passed_ = 0;
    for_all(trig_paths_, std::bind(&Path::clearCounters, _1));
    for_all(end_paths_, std::bind(&Path::clearCounters, _1));
    for_all(allWorkers(), std::bind(&Worker::clearCounters, _1));
  }

  void StreamSchedule::resetAll() {
    skippingEvent_ = false;
    results_->reset();
  }

  void StreamSchedule::addToAllWorkers(Worker* w) { workerManager_.addToAllWorkers(w); }

  void StreamSchedule::resetEarlyDelete() {
    //must be sure we have cleared the count first
    for (auto& count : earlyDeleteBranchToCount_) {
      count.count = 0;
    }
    //now reset based on how many helpers use that branch
    for (auto& index : earlyDeleteHelperToBranchIndicies_) {
      ++(earlyDeleteBranchToCount_[index].count);
    }
    for (auto& helper : earlyDeleteHelpers_) {
      helper.reset();
    }
  }

  void StreamSchedule::makePathStatusInserters(
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      ExceptionToActionTable const& actions) {
    int bitpos = 0;
    unsigned int indexEmpty = 0;
    unsigned int indexOfPath = 0;
    for (auto& pathStatusInserter : pathStatusInserters) {
      std::shared_ptr<PathStatusInserter> inserterPtr = get_underlying(pathStatusInserter);
      WorkerPtr workerPtr(
          new edm::WorkerT<PathStatusInserter::ModuleType>(inserterPtr, inserterPtr->moduleDescription(), &actions));
      pathStatusInserterWorkers_.emplace_back(workerPtr);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());

      // A little complexity here because a C++ Path object is not
      // instantiated and put into end_paths if there are no modules
      // on the configured path.
      if (indexEmpty < empty_trig_paths_.size() && bitpos == empty_trig_paths_.at(indexEmpty)) {
        ++indexEmpty;
      } else {
        trig_paths_.at(indexOfPath).setPathStatusInserter(inserterPtr.get(), workerPtr.get());
        ++indexOfPath;
      }
      ++bitpos;
    }

    bitpos = 0;
    indexEmpty = 0;
    indexOfPath = 0;
    for (auto& endPathStatusInserter : endPathStatusInserters) {
      std::shared_ptr<EndPathStatusInserter> inserterPtr = get_underlying(endPathStatusInserter);
      WorkerPtr workerPtr(
          new edm::WorkerT<EndPathStatusInserter::ModuleType>(inserterPtr, inserterPtr->moduleDescription(), &actions));
      endPathStatusInserterWorkers_.emplace_back(workerPtr);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());

      // A little complexity here because a C++ Path object is not
      // instantiated and put into end_paths if there are no modules
      // on the configured path.
      if (indexEmpty < empty_end_paths_.size() && bitpos == empty_end_paths_.at(indexEmpty)) {
        ++indexEmpty;
      } else {
        end_paths_.at(indexOfPath).setPathStatusInserter(nullptr, workerPtr.get());
        ++indexOfPath;
      }
      ++bitpos;
    }
  }
}  // namespace edm
