
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/src/esTaskArenas.h"
#include "tbb/global_control.h"

namespace edm {
  namespace {
    class ModuleBeginJobSignalSentry {
    public:
      ModuleBeginJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md) : a_(a), md_(&md) {
        if (a_)
          a_->preModuleBeginJobSignal_(*md_);
      }
      ~ModuleBeginJobSignalSentry() {
        if (a_)
          a_->postModuleBeginJobSignal_(*md_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleEndJobSignalSentry {
    public:
      ModuleEndJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md) : a_(a), md_(&md) {
        if (a_)
          a_->preModuleEndJobSignal_(*md_);
      }
      ~ModuleEndJobSignalSentry() {
        if (a_)
          a_->postModuleEndJobSignal_(*md_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleBeginStreamSignalSentry {
    public:
      ModuleBeginStreamSignalSentry(ActivityRegistry* a, StreamContext const& sc, ModuleCallingContext const& mcc)
          : a_(a), sc_(sc), mcc_(mcc) {
        if (a_)
          a_->preModuleBeginStreamSignal_(sc_, mcc_);
      }
      ~ModuleBeginStreamSignalSentry() {
        if (a_)
          a_->postModuleBeginStreamSignal_(sc_, mcc_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
    };

    class ModuleEndStreamSignalSentry {
    public:
      ModuleEndStreamSignalSentry(ActivityRegistry* a, StreamContext const& sc, ModuleCallingContext const& mcc)
          : a_(a), sc_(sc), mcc_(mcc) {
        if (a_)
          a_->preModuleEndStreamSignal_(sc_, mcc_);
      }
      ~ModuleEndStreamSignalSentry() {
        if (a_)
          a_->postModuleEndStreamSignal_(sc_, mcc_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
    };

  }  // namespace

  Worker::Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
      : timesRun_(0),
        timesVisited_(0),
        timesPassed_(0),
        timesFailed_(0),
        timesExcept_(0),
        state_(Ready),
        numberOfPathsOn_(0),
        numberOfPathsLeftToRun_(0),
        moduleCallingContext_(&iMD),
        actions_(iActions),
        cached_exception_(),
        actReg_(),
        earlyDeleteHelper_(nullptr),
        workStarted_(false),
        ranAcquireWithoutException_(false) {}

  Worker::~Worker() {}

  void Worker::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) { actReg_ = areg; }

  void Worker::exceptionContext(cms::Exception& ex, ModuleCallingContext const* mcc) {
    ModuleCallingContext const* imcc = mcc;
    while ((imcc->type() == ParentContext::Type::kModule) or (imcc->type() == ParentContext::Type::kInternal)) {
      std::ostringstream iost;
      if (imcc->state() == ModuleCallingContext::State::kPrefetching) {
        iost << "Prefetching for module ";
      } else {
        iost << "Calling method for module ";
      }
      iost << imcc->moduleDescription()->moduleName() << "/'" << imcc->moduleDescription()->moduleLabel() << "'";

      if (imcc->type() == ParentContext::Type::kInternal) {
        iost << " (probably inside some kind of mixing module)";
        imcc = imcc->internalContext()->moduleCallingContext();
      } else {
        imcc = imcc->moduleCallingContext();
      }
      ex.addContext(iost.str());
    }
    std::ostringstream ost;
    if (imcc->state() == ModuleCallingContext::State::kPrefetching) {
      ost << "Prefetching for module ";
    } else {
      ost << "Calling method for module ";
    }
    ost << imcc->moduleDescription()->moduleName() << "/'" << imcc->moduleDescription()->moduleLabel() << "'";
    ex.addContext(ost.str());

    if (imcc->type() == ParentContext::Type::kPlaceInPath) {
      ost.str("");
      ost << "Running path '";
      ost << imcc->placeInPathContext()->pathContext()->pathName() << "'";
      ex.addContext(ost.str());
      auto streamContext = imcc->placeInPathContext()->pathContext()->streamContext();
      if (streamContext) {
        ost.str("");
        edm::exceptionContext(ost, *streamContext);
        ex.addContext(ost.str());
      }
    } else {
      if (imcc->type() == ParentContext::Type::kStream) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->streamContext()));
        ex.addContext(ost.str());
      } else if (imcc->type() == ParentContext::Type::kGlobal) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->globalContext()));
        ex.addContext(ost.str());
      }
    }
  }

  bool Worker::shouldRethrowException(std::exception_ptr iPtr, ParentContext const& parentContext, bool isEvent) const {
    // NOTE: the warning printed as a result of ignoring or failing
    // a module will only be printed during the full true processing
    // pass of this module

    // Get the action corresponding to this exception.  However, if processing
    // something other than an event (e.g. run, lumi) always rethrow.
    if (not isEvent) {
      return true;
    }
    try {
      convertException::wrap([&]() { std::rethrow_exception(iPtr); });
    } catch (cms::Exception& ex) {
      exception_actions::ActionCodes action = actions_->find(ex.category());

      if (action == exception_actions::Rethrow) {
        return true;
      }

      ModuleCallingContext tempContext(description(), ModuleCallingContext::State::kInvalid, parentContext, nullptr);

      // If we are processing an endpath and the module was scheduled, treat SkipEvent or FailPath
      // as IgnoreCompletely, so any subsequent OutputModules are still run.
      // For unscheduled modules only treat FailPath as IgnoreCompletely but still allow SkipEvent to throw
      ModuleCallingContext const* top_mcc = tempContext.getTopModuleCallingContext();
      if (top_mcc->type() == ParentContext::Type::kPlaceInPath &&
          top_mcc->placeInPathContext()->pathContext()->isEndPath()) {
        if ((action == exception_actions::SkipEvent && tempContext.type() == ParentContext::Type::kPlaceInPath) ||
            action == exception_actions::FailPath) {
          action = exception_actions::IgnoreCompletely;
        }
      }
      if (action == exception_actions::IgnoreCompletely) {
        edm::printCmsExceptionWarning("IgnoreCompletely", ex);
        return false;
      }
    }
    return true;
  }

  void Worker::prePrefetchSelectionAsync(WaitingTask* successTask,
                                         ServiceToken const& token,
                                         StreamID id,
                                         EventPrincipal const* iPrincipal) {
    successTask->increment_ref_count();

    auto choiceTask = edm::make_waiting_task(
        tbb::task::allocate_root(), [id, successTask, iPrincipal, this, token](std::exception_ptr const*) {
          ServiceRegistry::Operate guard(token);
          // There is no reasonable place to rethrow, and implDoPrePrefetchSelection() should not throw in the first place.
          CMS_SA_ALLOW try {
            if (not implDoPrePrefetchSelection(id, *iPrincipal, &moduleCallingContext_)) {
              timesRun_.fetch_add(1, std::memory_order_relaxed);
              setPassed<true>();
              waitingTasks_.doneWaiting(nullptr);
              //TBB requires that destroyed tasks have count 0
              if (0 == successTask->decrement_ref_count()) {
                tbb::task::destroy(*successTask);
              }
              return;
            }
          } catch (...) {
          }
          if (0 == successTask->decrement_ref_count()) {
            tbb::task::spawn(*successTask);
          }
        });

    WaitingTaskHolder choiceHolder{choiceTask};

    std::vector<ProductResolverIndexAndSkipBit> items;
    itemsToGetForSelection(items);

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal->prefetchAsync(
            choiceHolder, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
    choiceHolder.doneWaiting(std::exception_ptr{});
  }

  void Worker::esPrefetchAsync(WaitingTaskHolder iTask,
                               EventSetupImpl const& iImpl,
                               Transition iTrans,
                               ServiceToken const& iToken) {
    if (iTrans >= edm::Transition::NumberOfEventSetupTransitions) {
      return;
    }
    auto const& recs = esRecordsToGetFrom(iTrans);
    auto const& items = esItemsToGetFrom(iTrans);

    assert(items.size() == recs.size());
    if (items.empty()) {
      return;
    }

    //Thread case of 1 thread special. The master thread is doing a wait_for_all on the
    // default tbb arena. It will not process any tasks on the es arena. We need to add a
    // task that will synchronously do a wait_for_all in the es arena to be sure prefetching
    // will work.

    if UNLIKELY (tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) == 1) {
      //We spawn this first so that the other ES tasks are before it in the TBB queue
      tbb::task_arena edArena(tbb::task_arena::attach{});
      tbb::task::spawn(*make_functor_task(
          tbb::task::allocate_root(), [this, task = std::move(iTask), iTrans, &iImpl, iToken, edArena]() mutable {
            esTaskArena().execute([this, iTrans, &iImpl, iToken, task = std::move(task), edArena]() {
              auto waitTask = edm::make_empty_waiting_task();
              auto const& recs = esRecordsToGetFrom(iTrans);
              auto const& items = esItemsToGetFrom(iTrans);
              waitTask->set_ref_count(2);
              {
                //Need hWaitTask to go out of scope before calling wait_for_all
                WaitingTaskHolder hWaitTask(waitTask.get());
                for (size_t i = 0; i != items.size(); ++i) {
                  if (recs[i] != ESRecordIndex{}) {
                    auto rec = iImpl.findImpl(recs[i]);
                    if (rec) {
                      rec->prefetchAsync(hWaitTask, items[i], &iImpl, iToken);
                    }
                  }
                }
              }
              waitTask->decrement_ref_count();
              waitTask->wait_for_all();

              auto exPtr = waitTask->exceptionPtr();
              tbb::task_arena(edArena).execute([task, exPtr]() {
                auto t = task;
                if (exPtr) {
                  t.doneWaiting(*exPtr);
                } else {
                  t.doneWaiting(std::exception_ptr{});
                }
              });
            });
          }));
    } else {
      //We need iTask to run in the default arena since it is not an ES task
      auto task = make_waiting_task(tbb::task::allocate_root(),
                                    [holder = std::move(iTask)](std::exception_ptr const* iExcept) mutable {
                                      if (iExcept) {
                                        holder.doneWaiting(*iExcept);
                                      } else {
                                        holder.doneWaiting(std::exception_ptr{});
                                      }
                                    });

      WaitingTaskHolder tempH(task);
      esTaskArena().execute([&]() {
        for (size_t i = 0; i != items.size(); ++i) {
          if (recs[i] != ESRecordIndex{}) {
            auto rec = iImpl.findImpl(recs[i]);
            if (rec) {
              rec->prefetchAsync(tempH, items[i], &iImpl, iToken);
            }
          }
        }
      });
    }
  }

  void Worker::edPrefetchAsync(WaitingTaskHolder iTask, ServiceToken const& token, Principal const& iPrincipal) const {
    // Prefetch products the module declares it consumes
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFrom(iPrincipal.branchType());

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
  }

  void Worker::setEarlyDeleteHelper(EarlyDeleteHelper* iHelper) { earlyDeleteHelper_ = iHelper; }

  void Worker::resetModuleDescription(ModuleDescription const* iDesc) {
    ModuleCallingContext temp(iDesc,
                              moduleCallingContext_.state(),
                              moduleCallingContext_.parent(),
                              moduleCallingContext_.previousModuleOnThread());
    moduleCallingContext_ = temp;
  }

  void Worker::beginJob() {
    try {
      convertException::wrap([&]() {
        ModuleBeginJobSignalSentry cpp(actReg_.get(), *description());
        implBeginJob();
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginJob for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::endJob() {
    try {
      convertException::wrap([&]() {
        ModuleDescription const* desc = description();
        assert(desc != nullptr);
        ModuleEndJobSignalSentry cpp(actReg_.get(), *desc);
        implEndJob();
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endJob for module " << description()->moduleName() << "/'" << description()->moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::beginStream(StreamID id, StreamContext& streamContext) {
    try {
      convertException::wrap([&]() {
        streamContext.setTransition(StreamContext::Transition::kBeginStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ParentContext parentContext(&streamContext);
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        moduleCallingContext_.setState(ModuleCallingContext::State::kRunning);
        ModuleBeginStreamSignalSentry beginSentry(actReg_.get(), streamContext, moduleCallingContext_);
        implBeginStream(id);
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginStream for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::endStream(StreamID id, StreamContext& streamContext) {
    try {
      convertException::wrap([&]() {
        streamContext.setTransition(StreamContext::Transition::kEndStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ParentContext parentContext(&streamContext);
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        moduleCallingContext_.setState(ModuleCallingContext::State::kRunning);
        ModuleEndStreamSignalSentry endSentry(actReg_.get(), streamContext, moduleCallingContext_);
        implEndStream(id);
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endStream for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper) {
    try {
      implRegisterThinnedAssociations(registry, helper);
    } catch (cms::Exception& ex) {
      ex.addContext("Calling registerThinnedAssociations() for module " + description()->moduleLabel());
      throw ex;
    }
  }

  void Worker::skipOnPath(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->pathFinished(iEvent);
    }
    if (0 == --numberOfPathsLeftToRun_) {
      waitingTasks_.doneWaiting(cached_exception_);
    }
  }

  void Worker::postDoEvent(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }

  void Worker::runAcquire(EventTransitionInfo const& info,
                          ParentContext const& parentContext,
                          WaitingTaskWithArenaHolder& holder) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    try {
      convertException::wrap([&]() { this->implDoAcquire(info, &moduleCallingContext_, holder); });
    } catch (cms::Exception& ex) {
      exceptionContext(ex, &moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, true)) {
        timesRun_.fetch_add(1, std::memory_order_relaxed);
        throw;
      }
    }
  }

  void Worker::runAcquireAfterAsyncPrefetch(std::exception_ptr const* iEPtr,
                                            EventTransitionInfo const& eventTransitionInfo,
                                            ParentContext const& parentContext,
                                            WaitingTaskWithArenaHolder holder) {
    ranAcquireWithoutException_ = false;
    std::exception_ptr exceptionPtr;
    if (iEPtr) {
      assert(*iEPtr);
      if (shouldRethrowException(*iEPtr, parentContext, true)) {
        exceptionPtr = *iEPtr;
      }
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    } else {
      // Caught exception is propagated via WaitingTaskWithArenaHolder
      CMS_SA_ALLOW try {
        runAcquire(eventTransitionInfo, parentContext, holder);
        ranAcquireWithoutException_ = true;
      } catch (...) {
        exceptionPtr = std::current_exception();
      }
    }
    // It is important this is after runAcquire completely finishes
    holder.doneWaiting(exceptionPtr);
  }

  std::exception_ptr Worker::handleExternalWorkException(std::exception_ptr const* iEPtr,
                                                         ParentContext const& parentContext) {
    if (ranAcquireWithoutException_) {
      try {
        convertException::wrap([iEPtr]() { std::rethrow_exception(*iEPtr); });
      } catch (cms::Exception& ex) {
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        exceptionContext(ex, &moduleCallingContext_);
        return std::current_exception();
      }
    }
    return *iEPtr;
  }

  Worker::HandleExternalWorkExceptionTask::HandleExternalWorkExceptionTask(Worker* worker,
                                                                           WaitingTask* runModuleTask,
                                                                           ParentContext const& parentContext)
      : m_worker(worker), m_runModuleTask(runModuleTask), m_parentContext(parentContext) {}

  tbb::task* Worker::HandleExternalWorkExceptionTask::execute() {
    auto excptr = exceptionPtr();
    if (excptr) {
      // increment the ref count so the holder will not spawn it
      m_runModuleTask->set_ref_count(1);
      WaitingTaskHolder holder(m_runModuleTask);
      holder.doneWaiting(m_worker->handleExternalWorkException(excptr, m_parentContext));
    }
    m_runModuleTask->set_ref_count(0);
    // Depend on TBB Scheduler Bypass to run the next task
    return m_runModuleTask;
  }
}  // namespace edm
