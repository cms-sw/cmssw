#include "FWCore/Framework/interface/maker/TransitionWorker_Event.h"
#include "FWCore/Framework/interface/EarlyDeleteHelper.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"

namespace edm {
  class TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::RunModuleTask : public WaitingTask {
  public:
    RunModuleTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                  EventTransitionInfo const& transitionInfo,
                  ServiceToken const& token,
                  StreamID streamID,
                  ParentContext const& parentContext,
                  StreamContext const* context,
                  oneapi::tbb::task_group* iGroup) noexcept
        : m_worker(worker),
          m_transitionInfo(transitionInfo),
          m_streamID(streamID),
          m_parentContext(parentContext),
          m_context(context),
          m_serviceToken(token),
          m_group(iGroup) {}

    void execute() final {
      //Need to make the services available early so other services can see them
      ServiceRegistry::Operate guard(m_serviceToken.lock());

      //incase the emit causes an exception, we need a memory location
      // to hold the exception_ptr
      std::exception_ptr temp_excptr;
      auto excptr = exceptionPtr();
      if (!m_worker->hasAcquire()) {
        // Caught exception is passed to TransitionWorkerBase::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskList
        CMS_SA_ALLOW try {
          //pre was called in prefetchAsync
          m_worker->emitPostModuleEventPrefetchingSignal();
        } catch (...) {
          temp_excptr = std::current_exception();
          if (not excptr) {
            excptr = temp_excptr;
          }
        }
      }

      if (not excptr) {
        if (auto queue = m_worker->serializeRunModule()) {
          auto f = [worker = m_worker,
                    info = m_transitionInfo,
                    streamID = m_streamID,
                    parentContext = m_parentContext,
                    sContext = m_context,
                    serviceToken = m_serviceToken]() {
            //Need to make the services available
            ServiceRegistry::Operate operateRunModule(serviceToken.lock());

            std::exception_ptr ptr;
            worker->runModuleAfterAsyncPrefetch(ptr, info, streamID, parentContext, sContext);
          };
          queue.push(*m_group, std::move(f));
          return;
        }
      }

      m_worker->runModuleAfterAsyncPrefetch(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
    }

  private:
    TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
    EventTransitionInfo m_transitionInfo;
    StreamID m_streamID;
    ParentContext const m_parentContext;
    StreamContext const* m_context;
    ServiceWeakToken m_serviceToken;
    oneapi::tbb::task_group* m_group;
  };

  class TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::AcquireTask : public WaitingTask {
  public:
    AcquireTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                EventTransitionInfo const& eventTransitionInfo,
                ServiceToken const& token,
                ParentContext const& parentContext,
                WaitingTaskHolder holder) noexcept
        : m_worker(worker),
          m_eventTransitionInfo(eventTransitionInfo),
          m_parentContext(parentContext),
          m_holder(std::move(holder)),
          m_serviceToken(token) {}

    void execute() final {
      //Need to make the services available early so other services can see them
      ServiceRegistry::Operate guard(m_serviceToken.lock());

      //incase the emit causes an exception, we need a memory location
      // to hold the exception_ptr
      std::exception_ptr temp_excptr;
      auto excptr = exceptionPtr();
      // Caught exception is passed to TransitionWorkerBase::runModuleAfterAsyncPrefetch(), which propagates it via WaitingTaskHolder
      CMS_SA_ALLOW try {
        //pre was called in prefetchAsync
        m_worker->emitPostModuleEventPrefetchingSignal();
      } catch (...) {
        temp_excptr = std::current_exception();
        if (not excptr) {
          excptr = temp_excptr;
        }
      }

      if (not excptr) {
        if (auto queue = m_worker->serializeRunModule()) {
          queue.push(*m_holder.group(),
                     [worker = m_worker,
                      info = m_eventTransitionInfo,
                      parentContext = m_parentContext,
                      serviceToken = m_serviceToken,
                      holder = std::move(m_holder)]() mutable {
                       //Need to make the services available
                       ServiceRegistry::Operate operateRunAcquire(serviceToken.lock());

                       std::exception_ptr ptr;
                       worker->runAcquireAfterAsyncPrefetch(ptr, info, parentContext, std::move(holder));
                     });
          return;
        }
      }

      m_worker->runAcquireAfterAsyncPrefetch(excptr, m_eventTransitionInfo, m_parentContext, std::move(m_holder));
    }

  private:
    TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
    EventTransitionInfo m_eventTransitionInfo;
    ParentContext const m_parentContext;
    WaitingTaskHolder m_holder;
    ServiceWeakToken m_serviceToken;
  };

  class TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::HandleExternalWorkExceptionTask
      : public WaitingTask {
  public:
    HandleExternalWorkExceptionTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                                    oneapi::tbb::task_group* group,
                                    WaitingTask* runModuleTask,
                                    ParentContext const& parentContext) noexcept;

    void execute() final;

  private:
    TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* m_worker;
    WaitingTask* m_runModuleTask;
    oneapi::tbb::task_group* m_group;
    ParentContext const m_parentContext;
  };

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::prePrefetchSelectionAsync(
      oneapi::tbb::task_group& group,
      WaitingTask* successTask,
      ServiceToken const& token,
      StreamID id,
      EventPrincipal const* iPrincipal) noexcept {
    successTask->increment_ref_count();

    ServiceWeakToken weakToken = token;
    auto choiceTask =
        edm::make_waiting_task([id, successTask, iPrincipal, this, weakToken, &group](std::exception_ptr const*) {
          ServiceRegistry::Operate guard(weakToken.lock());
          try {
            bool selected = convertException::wrap([&]() {
              if (not implDoPrePrefetchSelection(id, *iPrincipal, &moduleCallingContext_)) {
                setPassed();
                waitingTasks_.doneWaiting(nullptr);
                //TBB requires that destroyed tasks have count 0
                if (0 == successTask->decrement_ref_count()) {
                  TaskSentry s(successTask);
                }
                return false;
              }
              return true;
            });
            if (not selected) {
              return;
            }

          } catch (cms::Exception& e) {
            edm::exceptionContext(e, moduleCallingContext_);
            setException(std::current_exception());
            waitingTasks_.doneWaiting(std::current_exception());
            //TBB requires that destroyed tasks have count 0
            if (0 == successTask->decrement_ref_count()) {
              TaskSentry s(successTask);
            }
            return;
          }
          if (0 == successTask->decrement_ref_count()) {
            group.run([successTask]() {
              TaskSentry s(successTask);
              successTask->execute();
            });
          }
        });

    WaitingTaskHolder choiceHolder{group, choiceTask};

    std::vector<ProductResolverIndexAndSkipBit> items;
    itemsToGetForSelection(items);

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      if (productResolverIndex != ProductResolverIndexAmbiguous and
          productResolverIndex != ProductResolverIndexInvalid) {
        iPrincipal->prefetchAsync(choiceHolder, productResolverIndex, token, &moduleCallingContext_);
      }
    }
    choiceHolder.doneWaiting(std::exception_ptr{});
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::setEarlyDeleteHelper(EarlyDeleteHelper* iHelper) {
    earlyDeleteHelper_ = iHelper;
  }

  size_t TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::transformIndex(
      edm::ProductDescription const&) const noexcept {
    return -1;
  }
  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::doTransformAsync(WaitingTaskHolder iTask,
                                                                                      size_t iTransformIndex,
                                                                                      EventPrincipal const& iPrincipal,
                                                                                      ServiceToken const& iToken,
                                                                                      StreamID,
                                                                                      ModuleCallingContext const& mcc,
                                                                                      StreamContext const*) noexcept {
    ServiceWeakToken weakToken = iToken;

    //Need to make the services available early so other services can see them
    auto task = make_waiting_task(
        [this, iTask, weakToken, &iPrincipal, iTransformIndex, mcc](std::exception_ptr const* iExcept) mutable {
          //post prefetch signal
          actReg_->postModuleTransformPrefetchingSignal_.emit(*mcc.getStreamContext(), mcc);
          if (iExcept) {
            iTask.doneWaiting(*iExcept);
            return;
          }
          implDoTransformAsync(iTask, iTransformIndex, iPrincipal, mcc.parent(), weakToken);
        });

    //pre prefetch signal
    actReg_->preModuleTransformPrefetchingSignal_.emit(*mcc.getStreamContext(), mcc);
    iPrincipal.prefetchAsync(
        WaitingTaskHolder(*iTask.group(), task), itemToGetForTransform(iTransformIndex), iToken, &mcc);
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::skipOnPath(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->pathFinished(iEvent);
    }
    if (0 == --numberOfPathsLeftToRun_) {
      waitingTasks_.doneWaiting(cached_exception_);
    }
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::postDoEvent(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runAcquire(EventTransitionInfo const& info,
                                                                                ParentContext const& parentContext,
                                                                                WaitingTaskHolder holder) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    EventAcquireSignalsSentry sentry(activityRegistry(), &moduleCallingContext_);
    try {
      convertException::wrap([&]() { this->implDoAcquire(info, &moduleCallingContext_, std::move(holder)); });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      moduleCallingContext_.setState(ModuleCallingContext::State::kException);
      if (shouldRethrowException(std::current_exception(), parentContext, true, shouldTryToContinue_)) {
        throw;
      }
    }
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runAcquireAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      EventTransitionInfo const& eventTransitionInfo,
      ParentContext const& parentContext,
      WaitingTaskHolder holder) noexcept {
    ranAcquireWithoutException_ = false;
    std::exception_ptr exceptionPtr;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, true, shouldTryToContinue_)) {
        exceptionPtr = iEPtr;
      }
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    } else {
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try {
        // holder is copied to runAcquire in order to be independent
        // of the lifetime of the WaitingTaskHolder inside runAcquire
        runAcquire(eventTransitionInfo, parentContext, holder);
        ranAcquireWithoutException_ = true;
      } catch (...) {
        exceptionPtr = std::current_exception();
      }
    }
    // It is important this is after runAcquire completely finishes
    holder.doneWaiting(exceptionPtr);
  }

  std::exception_ptr TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::handleExternalWorkException(
      std::exception_ptr iEPtr, ParentContext const& parentContext) noexcept {
    if (ranAcquireWithoutException_) {
      try {
        convertException::wrap([iEPtr]() { std::rethrow_exception(iEPtr); });
      } catch (cms::Exception& ex) {
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        edm::exceptionContext(ex, moduleCallingContext_);
        return std::current_exception();
      }
    }
    return iEPtr;
  }

  TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::HandleExternalWorkExceptionTask::
      HandleExternalWorkExceptionTask(TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>* worker,
                                      oneapi::tbb::task_group* group,
                                      WaitingTask* runModuleTask,
                                      ParentContext const& parentContext) noexcept
      : m_worker(worker), m_runModuleTask(runModuleTask), m_group(group), m_parentContext(parentContext) {}

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::HandleExternalWorkExceptionTask::execute() {
    auto excptr = exceptionPtr();
    WaitingTaskHolder holder(*m_group, m_runModuleTask);
    if (excptr) {
      holder.doneWaiting(m_worker->handleExternalWorkException(excptr, m_parentContext));
    }
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::prefetchAsync(
      WaitingTaskHolder iTask,
      ServiceToken const& token,
      ParentContext const& parentContext,
      EventTransitionInfo const& transitionInfo,
      Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    actReg_->preModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);

    esPrefetchAsync(iTask, transitionInfo.eventSetupImpl(), iTransition, token);
    edPrefetchAsync(iTask, token, principal);

    preActionBeforeRunEventAsync(iTask, moduleCallingContext_, principal);
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl(
      WaitingTaskHolder task,
      EventTransitionInfo const& transitionInfo,
      ServiceToken const& token,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      //if have TriggerResults based selection we want to reject the event before doing prefetching
      if (needToRunSelection()) {
        //We need to run the selection in a different task so that
        // we can prefetch the data needed for the selection
        WaitingTask* moduleTask =
            new RunModuleTask(this, transitionInfo, token, streamID, parentContext, context, task.group());

        //make sure the task is either run or destroyed
        struct DestroyTask {
          DestroyTask(edm::WaitingTask* iTask) noexcept : m_task(iTask) {}

          ~DestroyTask() noexcept {
            auto p = m_task.exchange(nullptr);
            if (p) {
              TaskSentry s{p};
            }
          }

          edm::WaitingTask* release() noexcept { return m_task.exchange(nullptr); }

        private:
          std::atomic<edm::WaitingTask*> m_task;
        };
        if (hasAcquire()) {
          auto ownRunTask = std::make_shared<DestroyTask>(moduleTask);
          ServiceWeakToken weakToken = token;
          auto* group = task.group();
          moduleTask = make_waiting_task(
              [this, weakToken, transitionInfo, parentContext, ownRunTask, group](std::exception_ptr const* iExcept) {
                WaitingTaskHolder runTaskHolder(
                    *group, new HandleExternalWorkExceptionTask(this, group, ownRunTask->release(), parentContext));
                AcquireTask t(this, transitionInfo, weakToken.lock(), parentContext, runTaskHolder);
                t.execute();
              });
        }
        auto* group = task.group();
        auto ownModuleTask = std::make_shared<DestroyTask>(moduleTask);
        ServiceWeakToken weakToken = token;
        auto selectionTask =
            make_waiting_task([ownModuleTask, parentContext, info = transitionInfo, weakToken, group, this](
                                  std::exception_ptr const*) mutable {
              ServiceRegistry::Operate guard(weakToken.lock());
              prefetchAsync(WaitingTaskHolder(*group, ownModuleTask->release()),
                            weakToken.lock(),
                            parentContext,
                            info,
                            Transition::Event);
            });
        prePrefetchSelectionAsync(*group, selectionTask, token, streamID, &transitionInfo.principal());
      } else {
        WaitingTask* moduleTask =
            new RunModuleTask(this, transitionInfo, token, streamID, parentContext, context, task.group());
        auto group = task.group();
        if (hasAcquire()) {
          WaitingTaskHolder runTaskHolder(*group,
                                          new HandleExternalWorkExceptionTask(this, group, moduleTask, parentContext));
          moduleTask = new AcquireTask(this, transitionInfo, token, parentContext, std::move(runTaskHolder));
        }
        prefetchAsync(WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, Transition::Event);
      }
    }
  }

  std::exception_ptr TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      EventTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    std::exception_ptr exceptionPtr;
    bool shouldRun = true;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, true, shouldTryToContinue_)) {
        exceptionPtr = iEPtr;
        setException(exceptionPtr);
        shouldRun = false;
      } else {
        if (not shouldTryToContinue_) {
          setPassed();
          shouldRun = false;
        }
      }
    }
    if (shouldRun) {
      // Caught exception is propagated via WaitingTaskList
      CMS_SA_ALLOW try { runModule(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  void TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::doWorkNoEDPrefetchingAsyncImpl(
      WaitingTaskHolder task,
      EventTransitionInfo const& transitionInfo,
      ServiceToken const& serviceToken,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    auto workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);
    if (workStarted) {
      ServiceWeakToken weakToken = serviceToken;
      auto toDo = [this, info = transitionInfo, streamID, parentContext, context, weakToken]() {
        std::exception_ptr exceptionPtr;
        // Caught exception is propagated via WaitingTaskList
        CMS_SA_ALLOW try {
          //Need to make the services available
          ServiceRegistry::Operate guard(weakToken.lock());
          this->runModule(info, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };

      if (needsESPrefetching(Transition::Event)) {
        auto group = task.group();
        auto afterPrefetch =
            edm::make_waiting_task([toDo = std::move(toDo), group, this](std::exception_ptr const* iExcept) {
              if (iExcept) {
                this->waitingTasks_.doneWaiting(*iExcept);
              } else {
                if (auto queue = this->serializeRunModule()) {
                  queue.push(*group, toDo);
                } else {
                  group->run(toDo);
                }
              }
            });
        moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);
        esPrefetchAsync(
            WaitingTaskHolder(*group, afterPrefetch), transitionInfo.eventSetupImpl(), Transition::Event, serviceToken);
      } else {
        auto group = task.group();
        if (auto queue = this->serializeRunModule()) {
          queue.push(*group, toDo);
        } else {
          group->run(toDo);
        }
      }
    }
  }

  bool TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModule(EventTransitionInfo const& transitionInfo,
                                                                               StreamID streamID,
                                                                               ParentContext const& parentContext,
                                                                               StreamContext const* context) {
    //unscheduled producers should advance this
    //if (T::isEvent_) {
    //  ++timesVisited_;
    //}
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = call(streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed();
        } else {
          setFailed();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, true, shouldTryToContinue_)) {
        assert(not cached_exception_);
        setException(std::current_exception());
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed();
      }
    }

    return rc;
  }

  std::exception_ptr TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::runModuleDirectly(
      EventTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    std::exception_ptr prefetchingException;  // null because there was no prefetching to do
    return runModuleAfterAsyncPrefetch(prefetchingException, transitionInfo, streamID, parentContext, context);
  }

  bool TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>::call(StreamID,
                                                                          EventTransitionInfo const& info,
                                                                          ActivityRegistry* actReg,
                                                                          ModuleCallingContext* mcc,
                                                                          StreamContext const* context) {
    struct SignalTraits {
      using Context = StreamContext;
      static void preModuleSignal(ActivityRegistry* areg, Context const* context, ModuleCallingContext const* mcc) {
        areg->preModuleEventSignal_.emit(*context, *mcc);
      }
      static void postModuleSignal(ActivityRegistry* areg, Context const* context, ModuleCallingContext const* mcc) {
        areg->postModuleEventSignal_.emit(*context, *mcc);
      }
    };

    //Want postDoEvent to be called after signals are sent.
    auto postSentry = make_sentry(this, [&](auto* worker) { worker->postDoEvent(info.principal()); });
    ModuleSignalSentry<SignalTraits> signalSentry(actReg, context, mcc);
    signalSentry.preModuleSignal();
    bool returnValue;
    {
      ModuleCallingContextSentry mccSentry(*mcc);
      returnValue = this->implDo(info, mcc);
      mccSentry.finished(returnValue);
    }
    signalSentry.postModuleSignal();
    return returnValue;
  }
}  // namespace edm
