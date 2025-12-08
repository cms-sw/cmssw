#include "FWCore/Framework/interface/StreamSchedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/src/TriggerReport.h"
#include "FWCore/Framework/src/TriggerTimingReport.h"
#include "FWCore/Framework/src/ModuleHolderFactory.h"
#include "FWCore/Framework/interface/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/interface/WorkerInPath.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/ModuleRegistryUtilities.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"

#include "LuminosityBlockProcessingStatus.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <format>

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

    class BeginStreamTraits {
    public:
      static void preScheduleSignal(ActivityRegistry* activityRegistry, StreamContext const* streamContext) {
        activityRegistry->preBeginStreamSignal_.emit(*streamContext);
      }
      static void postScheduleSignal(ActivityRegistry* activityRegistry, StreamContext const* streamContext) {
        activityRegistry->postBeginStreamSignal_.emit(*streamContext);
      }
    };

    class EndStreamTraits {
    public:
      static void preScheduleSignal(ActivityRegistry* activityRegistry, StreamContext const* streamContext) {
        activityRegistry->preEndStreamSignal_.emit(*streamContext);
      }
      static void postScheduleSignal(ActivityRegistry* activityRegistry, StreamContext const* streamContext) {
        activityRegistry->postEndStreamSignal_.emit(*streamContext);
      }
    };

    // -----------------------------

    void initializeBranchToReadingWorker(std::vector<std::string> const& branchesToDeleteEarly,
                                         ProductRegistry const& preg,
                                         std::multimap<std::string, Worker*>& branchToReadingWorker) {
      auto vBranchesToDeleteEarly = branchesToDeleteEarly;
      // Remove any duplicates
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
  }  // namespace

  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  StreamSchedule::StreamSchedule(std::vector<PathInfo> const& paths,
                                 std::vector<EndPathInfo> const& endpaths,
                                 std::vector<ModuleDescription const*> const& unscheduledModules,
                                 std::shared_ptr<TriggerResultInserter> inserter,
                                 std::shared_ptr<ModuleRegistry> modReg,
                                 ExceptionToActionTable const& actions,
                                 std::shared_ptr<ActivityRegistry> areg,
                                 StreamID streamID,
                                 ProcessContext const* processContext)
      : workerManagerRuns_(modReg, areg, actions),
        workerManagerLumisAndEvents_(modReg, areg, actions),
        actReg_(areg),
        results_(std::make_shared<HLTGlobalStatus>(paths.size())),
        results_inserter_(),
        trig_paths_(),
        end_paths_(),
        total_events_(),
        total_passed_(),
        number_of_unscheduled_modules_(0),
        streamID_(streamID),
        streamContext_(streamID_, processContext) {
    bool hasPath = false;

    int trig_bitpos = 0;
    trig_paths_.reserve(paths.size());
    for (auto const& path : paths) {
      fillTrigPath(path, trig_bitpos, results());
      ++trig_bitpos;
      hasPath = true;
    }

    if (hasPath) {
      // the results inserter stands alone
      inserter->setTrigResultForStream(streamID.value(), results());
      results_inserter_ = workerManagerLumisAndEvents_.getWorkerForModule(*inserter);
    }

    // fill normal endpaths
    int bitpos = 0;
    end_paths_.reserve(endpaths.size());
    for (auto const& end_path : endpaths) {
      fillEndPath(end_path, bitpos);
      ++bitpos;
    }

    for (auto const* module : unscheduledModules) {
      workerManagerLumisAndEvents_.addToUnscheduledWorkers(*module);
    }

    for (auto const& worker : allWorkersLumisAndEvents()) {
      (void)workerManagerRuns_.getWorkerForModule(*worker->description());
    }

  }  // StreamSchedule::StreamSchedule

  void StreamSchedule::initializeEarlyDelete(ModuleRegistry& modReg,
                                             std::vector<std::string> const& branchesToDeleteEarly,
                                             std::multimap<std::string, std::string> const& referencesToBranches,
                                             std::vector<std::string> const& modulesToSkip,
                                             edm::ProductRegistry const& preg) {
    // setup the list with those products actually registered for this job
    std::multimap<std::string, Worker*> branchToReadingWorker;
    initializeBranchToReadingWorker(branchesToDeleteEarly, preg, branchToReadingWorker);

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
            ProductDescription const& desc = *item.first;
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

    std::unordered_set<std::string> modulesToExclude(modulesToSkip.begin(), modulesToSkip.end());
    for (auto w : allWorkersLumisAndEvents()) {
      if (modulesToExclude.end() != modulesToExclude.find(w->description()->moduleLabel())) {
        continue;
      }
      //determine if this module could read a branch we want to delete early
      auto consumes = modReg.getExistingModule(w->description()->moduleLabel())->moduleConsumesInfos();
      if (not consumes.empty()) {
        bool foundAtLeastOneMatchingBranch = false;
        for (auto const& product : consumes) {
          std::string branch = std::format("{}_{}_{}_{}",
                                           product.type().friendlyClassName(),
                                           product.label().data(),
                                           product.instance().data(),
                                           product.process().data());
          {
            //Handle case where worker directly consumes product
            auto found = branchToReadingWorker.end();
            if (product.process().empty()) {
              auto startFound = branchToReadingWorker.lower_bound(branch);
              if (startFound != branchToReadingWorker.end()) {
                if (startFound->first.substr(0, branch.size()) == branch) {
                  //match all processNames here, even if it means multiple matches will happen
                  found = startFound;
                }
              }
            } else {
              auto exactFound = branchToReadingWorker.equal_range(branch);
              if (exactFound.first != exactFound.second) {
                found = exactFound.first;
              }
            }
            if (found != branchToReadingWorker.end()) {
              if (not foundAtLeastOneMatchingBranch) {
                ++upperLimitOnReadingWorker;
                foundAtLeastOneMatchingBranch = true;
              }
              ++upperLimitOnIndicies;
              ++reserveSizeForWorker[w];
              if (nullptr == found->second) {
                found->second = w;
              } else {
                branchToReadingWorker.insert(make_pair(found->first, w));
              }
            }
          }
          {
            //Handle case where indirectly consumes product
            auto found = referencesToBranches.end();
            if (product.process().empty()) {
              auto startFound = referencesToBranches.lower_bound(branch);
              if (startFound != referencesToBranches.end()) {
                if (startFound->first.substr(0, branch.size()) == branch) {
                  //match all processNames here, even if it means multiple matches will happen
                  found = startFound;
                }
              }
            } else {
              //can match exactly
              auto exactFound = referencesToBranches.equal_range(branch);
              if (exactFound.first != exactFound.second) {
                found = exactFound.first;
              }
            }
            if (found != referencesToBranches.end()) {
              for (auto itr = found; (itr != referencesToBranches.end()) and (itr->first == found->first); ++itr) {
                auto foundInBranchToReadingWorker = branchToReadingWorker.find(itr->second);
                if (foundInBranchToReadingWorker == branchToReadingWorker.end()) {
                  continue;
                }
                if (not foundAtLeastOneMatchingBranch) {
                  ++upperLimitOnReadingWorker;
                  foundAtLeastOneMatchingBranch = true;
                }
                ++upperLimitOnIndicies;
                ++reserveSizeForWorker[w];
                if (nullptr == foundInBranchToReadingWorker->second) {
                  foundInBranchToReadingWorker->second = w;
                } else {
                  branchToReadingWorker.insert(make_pair(itr->second, w));
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
      if (not unusedBranches.empty() and streamID_.value() == 0) {
        LogWarning l("UnusedProductsForCanDeleteEarly");
        l << "The following products in the 'canDeleteEarly' list are not used in this job and will be ignored.\n"
             " If possible, remove the producer from the job.";
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
          assert(index < earlyDeleteHelperToBranchIndicies_.size());
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

  StreamSchedule::PathWorkers StreamSchedule::fillWorkers(std::vector<ModuleInPath> const& iPath) {
    PathWorkers tmpworkers;
    tmpworkers.reserve(iPath.size());
    for (auto const& module : iPath) {
      tmpworkers.emplace_back(workerManagerLumisAndEvents_.getWorkerForModule(*module.description_),
                              module.action_,
                              module.placeInPath_,
                              module.runConcurrently_);
    }
    return tmpworkers;
  }

  void StreamSchedule::fillTrigPath(PathInfo const& iPath, int bitpos, TrigResPtr trptr) {
    auto workerPtr = workerManagerLumisAndEvents_.getWorkerForModule(*iPath.inserter_);
    pathStatusInserterWorkers_.emplace_back(workerPtr);
    if (iPath.modules_.empty()) {
      empty_trig_paths_.push_back(bitpos);
    } else {
      auto tmpworkers = fillWorkers(iPath.modules_);
      trig_paths_.emplace_back(
          bitpos, iPath.name_, tmpworkers, trptr, actionTable(), actReg_, &streamContext_, PathContext::PathType::kPath);
      trig_paths_.back().setPathStatusInserter(iPath.inserter_.get(), workerPtr);
    }
  }

  void StreamSchedule::fillEndPath(EndPathInfo const& iEndPath, int bitpos) {
    Worker* workerPtr = nullptr;
    if (iEndPath.inserter_) {
      workerPtr = workerManagerLumisAndEvents_.getWorkerForModule(*iEndPath.inserter_);
      endPathStatusInserterWorkers_.emplace_back(workerPtr);
    }
    if (iEndPath.modules_.empty()) {
      empty_end_paths_.push_back(bitpos);
    } else {
      PathWorkers tmpworkers = fillWorkers(iEndPath.modules_);
      end_paths_.emplace_back(bitpos,
                              iEndPath.name_,
                              tmpworkers,
                              TrigResPtr(),
                              actionTable(),
                              actReg_,
                              &streamContext_,
                              PathContext::PathType::kEndPath);
      if (iEndPath.inserter_) {
        end_paths_.back().setPathStatusInserter(nullptr, workerPtr);
      }
    }
  }

  void StreamSchedule::beginStream(ModuleRegistry& iModuleRegistry) {
    streamContext_.setTransition(StreamContext::Transition::kBeginStream);
    streamContext_.setEventID(EventID(0, 0, 0));
    streamContext_.setRunIndex(RunIndex::invalidRunIndex());
    streamContext_.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
    streamContext_.setTimestamp(Timestamp());

    std::exception_ptr exceptionInStream;
    CMS_SA_ALLOW try {
      preScheduleSignal<BeginStreamTraits>(&streamContext_);
      runBeginStreamForModules(streamContext_, iModuleRegistry, *actReg_, moduleBeginStreamFailed_);
    } catch (...) {
      exceptionInStream = std::current_exception();
    }

    postScheduleSignal<BeginStreamTraits>(&streamContext_, exceptionInStream);

    if (exceptionInStream) {
      bool cleaningUpAfterException = false;
      handleException(streamContext_, cleaningUpAfterException, exceptionInStream);
    }
    streamContext_.setTransition(StreamContext::Transition::kInvalid);

    if (exceptionInStream) {
      std::rethrow_exception(exceptionInStream);
    }
  }

  void StreamSchedule::endStream(ModuleRegistry& iModuleRegistry,
                                 ExceptionCollector& collector,
                                 std::mutex& collectorMutex) noexcept {
    streamContext_.setTransition(StreamContext::Transition::kEndStream);
    streamContext_.setEventID(EventID(0, 0, 0));
    streamContext_.setRunIndex(RunIndex::invalidRunIndex());
    streamContext_.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
    streamContext_.setTimestamp(Timestamp());

    std::exception_ptr exceptionInStream;
    CMS_SA_ALLOW try {
      preScheduleSignal<EndStreamTraits>(&streamContext_);
      runEndStreamForModules(
          streamContext_, iModuleRegistry, *actReg_, collector, collectorMutex, moduleBeginStreamFailed_);
    } catch (...) {
      exceptionInStream = std::current_exception();
    }

    postScheduleSignal<EndStreamTraits>(&streamContext_, exceptionInStream);

    if (exceptionInStream) {
      std::lock_guard<std::mutex> collectorLock(collectorMutex);
      collector.call([&exceptionInStream]() { std::rethrow_exception(exceptionInStream); });
    }
    streamContext_.setTransition(StreamContext::Transition::kInvalid);
  }

  void StreamSchedule::replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel) {
    for (auto const& worker : allWorkersRuns()) {
      if (worker->description()->moduleLabel() == iLabel) {
        iMod->replaceModuleFor(worker);
        try {
          convertException::wrap([&] { iMod->beginStream(streamID_); });
        } catch (cms::Exception& ex) {
          moduleBeginStreamFailed_.emplace_back(iMod->moduleDescription().id());
          ex.addContext("Executing StreamSchedule::replaceModule");
          throw;
        }
        break;
      }
    }

    for (auto const& worker : allWorkersLumisAndEvents()) {
      if (worker->description()->moduleLabel() == iLabel) {
        iMod->replaceModuleFor(worker);
        break;
      }
    }
  }

  void StreamSchedule::deleteModule(std::string const& iLabel) {
    workerManagerRuns_.deleteModuleIfExists(iLabel);
    workerManagerLumisAndEvents_.deleteModuleIfExists(iLabel);
  }

  std::vector<ModuleDescription const*> StreamSchedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkersLumisAndEvents().size());

    for (auto const& worker : allWorkersLumisAndEvents()) {
      ModuleDescription const* p = worker->description();
      result.push_back(p);
    }
    return result;
  }

  void StreamSchedule::processOneEventAsync(
      WaitingTaskHolder iTask,
      EventTransitionInfo& info,
      ServiceToken const& serviceToken,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters) {
    EventPrincipal& ep = info.principal();

    // Caught exception is propagated via WaitingTaskHolder
    CMS_SA_ALLOW try {
      this->resetAll();

      using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

      Traits::setStreamContext(streamContext_, ep);
      //a service may want to communicate with another service
      ServiceRegistry::Operate guard(serviceToken);
      Traits::preScheduleSignal(actReg_.get(), &streamContext_);

      // Data dependencies need to be set up before marking empty
      // (End)Paths complete in case something consumes the status of
      // the empty (EndPath)
      workerManagerLumisAndEvents_.setupResolvers(ep);
      workerManagerLumisAndEvents_.setupOnDemandSystem(info);

      HLTPathStatus hltPathStatus(hlt::Pass, 0);
      for (int empty_trig_path : empty_trig_paths_) {
        results_->at(empty_trig_path) = hltPathStatus;
        pathStatusInserters[empty_trig_path]->setPathStatus(streamID_, hltPathStatus);
        std::exception_ptr except = pathStatusInserterWorkers_[empty_trig_path]
                                        ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                                            info, streamID_, ParentContext(&streamContext_), &streamContext_);
        if (except) {
          iTask.doneWaiting(except);
          return;
        }
      }
      if (not endPathStatusInserterWorkers_.empty()) {
        for (int empty_end_path : empty_end_paths_) {
          std::exception_ptr except =
              endPathStatusInserterWorkers_[empty_end_path]
                  ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                      info, streamID_, ParentContext(&streamContext_), &streamContext_);
          if (except) {
            iTask.doneWaiting(except);
            return;
          }
        }
      }

      ++total_events_;

      //use to give priorities on an error to ones from Paths
      auto pathErrorHolder = std::make_unique<std::atomic<std::exception_ptr*>>(nullptr);
      auto pathErrorPtr = pathErrorHolder.get();
      ServiceWeakToken weakToken = serviceToken;
      auto allPathsDone = make_waiting_task(
          [iTask, this, weakToken, pathError = std::move(pathErrorHolder)](std::exception_ptr const* iPtr) mutable {
            ServiceRegistry::Operate operate(weakToken.lock());

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
      WaitingTaskHolder allPathsHolder(*iTask.group(), allPathsDone);

      auto pathsDone = make_waiting_task([allPathsHolder, pathErrorPtr, transitionInfo = info, this, weakToken](
                                             std::exception_ptr const* iPtr) mutable {
        ServiceRegistry::Operate operate(weakToken.lock());

        if (iPtr) {
          // free previous value of pathErrorPtr, if any;
          // prioritize this error over one that happens in EndPath or Accumulate
          auto currentPtr = pathErrorPtr->exchange(new std::exception_ptr(*iPtr));
          assert(currentPtr == nullptr);
        }
        finishedPaths(*pathErrorPtr, std::move(allPathsHolder), transitionInfo);
      });

      //The holder guarantees that if the paths finish before the loop ends
      // that we do not start too soon. It also guarantees that the task will
      // run under that condition.
      WaitingTaskHolder taskHolder(*iTask.group(), pathsDone);

      //start end paths first so on single threaded the paths will run first
      WaitingTaskHolder hAllPathsDone(*iTask.group(), allPathsDone);
      for (auto it = end_paths_.rbegin(), itEnd = end_paths_.rend(); it != itEnd; ++it) {
        it->processEventUsingPathAsync(hAllPathsDone, info, serviceToken, streamID_, &streamContext_);
      }

      for (auto it = trig_paths_.rbegin(), itEnd = trig_paths_.rend(); it != itEnd; ++it) {
        it->processEventUsingPathAsync(taskHolder, info, serviceToken, streamID_, &streamContext_);
      }

      ParentContext parentContext(&streamContext_);
      workerManagerLumisAndEvents_.processAccumulatorsAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
          hAllPathsDone, info, serviceToken, streamID_, parentContext, &streamContext_);
    } catch (...) {
      iTask.doneWaiting(std::current_exception());
    }
  }

  void StreamSchedule::finishedPaths(std::atomic<std::exception_ptr*>& iExcept,
                                     WaitingTaskHolder iWait,
                                     EventTransitionInfo& info) {
    if (iExcept) {
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try { std::rethrow_exception(*(iExcept.load())); } catch (cms::Exception& e) {
        exception_actions::ActionCodes action = actionTable().find(e.category());
        assert(action != exception_actions::IgnoreCompletely);
        if (action == exception_actions::TryToContinue) {
          edm::printCmsExceptionWarning("TryToContinue", e);
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

        auto expt = results_inserter_->runModuleDirectly<Traits>(info, streamID_, parentContext, &streamContext_);
        if (expt) {
          std::rethrow_exception(expt);
        }
      } catch (cms::Exception& ex) {
        if (not iExcept) {
          if (ex.context().empty()) {
            std::ostringstream ost;
            ost << "Processing Event " << info.principal().id();
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

      actReg_->preStreamEarlyTerminationSignal_.emit(streamContext_, TerminationOrigin::ExceptionFromThisContext);
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
        oLabelsToFill.push_back(itFound->getWorker(i)->description()->moduleLabel());
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
        descriptions.push_back(itFound->getWorker(i)->description());
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
        descriptions.push_back(itFound->getWorker(i)->description());
      }
    }
  }

  static void fillModuleInPathSummary(Path const& path, size_t which, ModuleInPathSummary& sum) {
    sum.timesVisited += path.timesVisited(which);
    sum.timesPassed += path.timesPassed(which);
    sum.timesFailed += path.timesFailed(which);
    sum.timesExcept += path.timesExcept(which);
    sum.moduleLabel = path.getWorker(which)->description()->moduleLabel();
    sum.bitPosition = path.bitPosition(which);
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
    sum.moduleLabel = w.description()->moduleLabel();
  }

  static void fillWorkerSummary(Worker const* pw, WorkerSummary& sum) { fillWorkerSummaryAux(*pw, sum); }

  void StreamSchedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents += totalEvents();
    rep.eventSummary.totalEventsPassed += totalEventsPassed();
    rep.eventSummary.totalEventsFailed += totalEventsFailed();

    fill_summary(trig_paths_, rep.trigPathSummaries, &fillPathSummary);
    fill_summary(end_paths_, rep.endPathSummaries, &fillPathSummary);
    fill_summary(allWorkersLumisAndEvents(), rep.workerSummaries, &fillWorkerSummary);
  }

  void StreamSchedule::clearCounters() {
    using std::placeholders::_1;
    total_events_ = total_passed_ = 0;
    for_all(trig_paths_, std::bind(&Path::clearCounters, _1));
    for_all(end_paths_, std::bind(&Path::clearCounters, _1));
    for_all(allWorkersLumisAndEvents(), std::bind(&Worker::clearCounters, _1));
  }

  void StreamSchedule::resetAll() { results_->reset(); }

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

  void StreamSchedule::handleException(StreamContext const& streamContext,
                                       bool cleaningUpAfterException,
                                       std::exception_ptr& excpt) const noexcept {
    //add context information to the exception and print message
    try {
      convertException::wrap([&excpt]() { std::rethrow_exception(excpt); });
    } catch (cms::Exception& ex) {
      std::ostringstream ost;
      // In most cases the exception will already have context at this point,
      // but add some context here in those rare cases where it does not.
      if (ex.context().empty()) {
        exceptionContext(ost, streamContext);
      }
      addContextAndPrintException(ost.str().c_str(), ex, cleaningUpAfterException);
      excpt = std::current_exception();
    }
    // We are already handling an earlier exception, so ignore it
    // if this signal results in another exception being thrown.
    CMS_SA_ALLOW try {
      actReg_->preStreamEarlyTerminationSignal_.emit(streamContext, TerminationOrigin::ExceptionFromThisContext);
    } catch (...) {
    }
  }
}  // namespace edm
