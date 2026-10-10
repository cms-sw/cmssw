#include "FWCore/Framework/interface/maker/TransitionWorker_Stream.h"

namespace edm {
  namespace workerhelper {
    template <>
    class CallStreamImpl<RunTransitionInfo, TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>*,
                       StreamID,
                       RunTransitionInfo const&,
                       ActivityRegistry*,
                       ModuleCallingContext const*,
                       StreamContext const*);
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  RunTransitionInfo const&,
                                  Transition) noexcept;
    };

    template <>
    class CallStreamImpl<RunTransitionInfo, TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>*,
                       StreamID,
                       RunTransitionInfo const&,
                       ActivityRegistry*,
                       ModuleCallingContext const*,
                       StreamContext const*);
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  RunTransitionInfo const&,
                                  Transition) noexcept;
    };

    template <>
    class CallStreamImpl<LumiTransitionInfo, TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>*,
                       StreamID,
                       LumiTransitionInfo const&,
                       ActivityRegistry*,
                       ModuleCallingContext const*,
                       StreamContext const*);
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  LumiTransitionInfo const&,
                                  Transition) noexcept;
    };

    template <>
    class CallStreamImpl<LumiTransitionInfo, TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>*,
                       StreamID,
                       LumiTransitionInfo const&,
                       ActivityRegistry*,
                       ModuleCallingContext const*,
                       StreamContext const*);
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>*,
                                  WaitingTaskHolder,
                                  ServiceToken const&,
                                  LumiTransitionInfo const&,
                                  Transition) noexcept;
    };
  }  // namespace workerhelper

  template <typename TI>
  template <TransitionEdge E>
  class TransitionWorker<TI, TransitionPhaseStream>::RunModuleTask : public WaitingTask {
  public:
    RunModuleTask(TransitionWorker<TI, TransitionPhaseStream>* worker,
                  TI const& transitionInfo,
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
      m_worker->emitPostModuleStreamPrefetchingSignal();

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
            worker->template runModuleAfterAsyncPrefetch<E>(ptr, info, streamID, parentContext, sContext);
          };
          //keep another global transition from running if necessary
          auto gQueue = workerhelper::CallStreamImpl<TI, E>::pauseGlobalQueue(m_worker);
          if (gQueue) {
            gQueue->push(*m_group, [queue, gQueue, f, group = m_group]() mutable {
              gQueue->pause();
              queue.push(*group, std::move(f));
            });
          } else {
            queue.push(*m_group, std::move(f));
          }
          return;
        }
      }

      m_worker->runModuleAfterAsyncPrefetch<E>(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
    }

  private:
    TransitionWorker<TI, TransitionPhaseStream>* m_worker;
    TI m_transitionInfo;
    StreamID m_streamID;
    ParentContext const m_parentContext;
    StreamContext const* m_context;
    ServiceWeakToken m_serviceToken;
    oneapi::tbb::task_group* m_group;
  };

  namespace workerhelper {
    bool CallStreamImpl<RunTransitionInfo, TransitionEdge::kBegin>::call(
        TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
        StreamID id,
        RunTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        StreamContext const* context) {
      struct SignalTrait {
        using Context = StreamContext;
        static void preModuleSignal(ActivityRegistry* a,
                                    StreamContext const* streamContext,
                                    ModuleCallingContext const* moduleCallingContext) {
          a->preModuleStreamBeginRunSignal_.emit(*streamContext, *moduleCallingContext);
        }
        static void postModuleSignal(ActivityRegistry* a,
                                     StreamContext const* streamContext,
                                     ModuleCallingContext const* moduleCallingContext) {
          a->postModuleStreamBeginRunSignal_.emit(*streamContext, *moduleCallingContext);
        }
      };
      ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
      cpp.preModuleSignal();
      auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
      cpp.postModuleSignal();
      iWorker->beginSucceeded_ = true;
      return returnValue;
    }

    void CallStreamImpl<RunTransitionInfo, TransitionEdge::kBegin>::esPrefetchAsync(
        TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
        WaitingTaskHolder waitingTask,
        ServiceToken const& token,
        RunTransitionInfo const& info,
        Transition transition) noexcept {
      worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
    }

    bool CallStreamImpl<RunTransitionInfo, TransitionEdge::kEnd>::call(
        TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
        StreamID id,
        RunTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        StreamContext const* context) {
      if (iWorker->beginSucceeded_) {
        iWorker->beginSucceeded_ = false;

        struct SignalTrait {
          using Context = StreamContext;
          static void preModuleSignal(ActivityRegistry* a,
                                      StreamContext const* streamContext,
                                      ModuleCallingContext const* moduleCallingContext) {
            a->preModuleStreamEndRunSignal_.emit(*streamContext, *moduleCallingContext);
          }
          static void postModuleSignal(ActivityRegistry* a,
                                       StreamContext const* streamContext,
                                       ModuleCallingContext const* moduleCallingContext) {
            a->postModuleStreamEndRunSignal_.emit(*streamContext, *moduleCallingContext);
          }
        };
        ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
        cpp.postModuleSignal();
        return returnValue;
      }
      return true;
    }

    void CallStreamImpl<RunTransitionInfo, TransitionEdge::kEnd>::esPrefetchAsync(
        TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
        WaitingTaskHolder waitingTask,
        ServiceToken const& token,
        RunTransitionInfo const& info,
        Transition transition) noexcept {
      worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
    }

    bool CallStreamImpl<LumiTransitionInfo, TransitionEdge::kBegin>::call(
        TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
        StreamID id,
        LumiTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        StreamContext const* context) {
      struct SignalTrait {
        using Context = StreamContext;
        static void preModuleSignal(ActivityRegistry* a,
                                    StreamContext const* streamContext,
                                    ModuleCallingContext const* moduleCallingContext) {
          a->preModuleStreamBeginLumiSignal_.emit(*streamContext, *moduleCallingContext);
        }
        static void postModuleSignal(ActivityRegistry* a,
                                     StreamContext const* streamContext,
                                     ModuleCallingContext const* moduleCallingContext) {
          a->postModuleStreamBeginLumiSignal_.emit(*streamContext, *moduleCallingContext);
        }
      };
      ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
      cpp.preModuleSignal();
      auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
      cpp.postModuleSignal();
      iWorker->beginSucceeded_ = true;
      return returnValue;
    }

    void CallStreamImpl<LumiTransitionInfo, TransitionEdge::kBegin>::esPrefetchAsync(
        TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
        WaitingTaskHolder waitingTask,
        ServiceToken const& token,
        LumiTransitionInfo const& info,
        Transition transition) noexcept {
      worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
    }

    bool CallStreamImpl<LumiTransitionInfo, TransitionEdge::kEnd>::call(
        TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
        StreamID id,
        LumiTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        StreamContext const* context) {
      if (iWorker->beginSucceeded_) {
        iWorker->beginSucceeded_ = false;

        struct SignalTrait {
          using Context = StreamContext;
          static void preModuleSignal(ActivityRegistry* a,
                                      StreamContext const* streamContext,
                                      ModuleCallingContext const* moduleCallingContext) {
            a->preModuleStreamEndLumiSignal_.emit(*streamContext, *moduleCallingContext);
          }
          static void postModuleSignal(ActivityRegistry* a,
                                       StreamContext const* streamContext,
                                       ModuleCallingContext const* moduleCallingContext) {
            a->postModuleStreamEndLumiSignal_.emit(*streamContext, *moduleCallingContext);
          }
        };
        ModuleSignalSentry<SignalTrait> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
        cpp.postModuleSignal();
        return returnValue;
      }
      return true;
    }

    void CallStreamImpl<LumiTransitionInfo, TransitionEdge::kEnd>::esPrefetchAsync(
        TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
        WaitingTaskHolder waitingTask,
        ServiceToken const& token,
        LumiTransitionInfo const& info,
        Transition transition) noexcept {
      worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
    }
  }  // namespace workerhelper

  template <typename TI>
  template <TransitionEdge E>
  std::exception_ptr TransitionWorker<TI, TransitionPhaseStream>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      TI const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      StreamContext const* context) noexcept {
    std::exception_ptr exceptionPtr;
    bool shouldRun = true;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, false, shouldTryToContinue_)) {
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
      CMS_SA_ALLOW try { runModule<E>(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  template <typename TI>
  template <TransitionEdge E>
  void TransitionWorker<TI, TransitionPhaseStream>::doWorkNoEDPrefetchingAsyncImpl(
      WaitingTaskHolder task,
      TI const& transitionInfo,
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

          this->runModule<E>(info, streamID, parentContext, context);
        } catch (...) {
          exceptionPtr = std::current_exception();
        }
        this->waitingTasks_.doneWaiting(exceptionPtr);
      };

      if (needsESPrefetching(TransitionTrait<TI, E>::value)) {
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
        esPrefetchAsync(WaitingTaskHolder(*group, afterPrefetch),
                        transitionInfo.eventSetupImpl(),
                        TransitionTrait<TI, E>::value,
                        serviceToken);
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

  template <typename TI>
  template <TransitionEdge E>
  bool TransitionWorker<TI, TransitionPhaseStream>::runModule(TI const& transitionInfo,
                                                              StreamID streamID,
                                                              ParentContext const& parentContext,
                                                              StreamContext const* context) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallStreamImpl<TI, E>::call(
            this, streamID, transitionInfo, actReg_.get(), &moduleCallingContext_, context);

        if (rc) {
          setPassed();
        } else {
          setFailed();
        }
      });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, false, shouldTryToContinue_)) {
        assert(not cached_exception_);
        setException(std::current_exception());
        std::rethrow_exception(cached_exception_);
      } else {
        rc = setPassed();
      }
    }

    return rc;
  }

  template void TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::doWorkNoEDPrefetchingAsyncImpl<
      TransitionEdge::kBegin>(WaitingTaskHolder,
                              RunTransitionInfo const&,
                              ServiceToken const&,
                              StreamID,
                              ParentContext const&,
                              StreamContext const*) noexcept;
  template void TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::doWorkNoEDPrefetchingAsyncImpl<
      TransitionEdge::kEnd>(WaitingTaskHolder,
                            RunTransitionInfo const&,
                            ServiceToken const&,
                            StreamID,
                            ParentContext const&,
                            StreamContext const*) noexcept;
  template void TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::doWorkNoEDPrefetchingAsyncImpl<
      TransitionEdge::kBegin>(WaitingTaskHolder,
                              LumiTransitionInfo const&,
                              ServiceToken const&,
                              StreamID,
                              ParentContext const&,
                              StreamContext const*) noexcept;
  template void TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::doWorkNoEDPrefetchingAsyncImpl<
      TransitionEdge::kEnd>(WaitingTaskHolder,
                            LumiTransitionInfo const&,
                            ServiceToken const&,
                            StreamID,
                            ParentContext const&,
                            StreamContext const*) noexcept;
  template std::exception_ptr
  TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::runModuleAfterAsyncPrefetch<TransitionEdge::kBegin>(
      std::exception_ptr, RunTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;
  template std::exception_ptr
  TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::runModuleAfterAsyncPrefetch<TransitionEdge::kEnd>(
      std::exception_ptr, RunTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;
  template std::exception_ptr
  TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::runModuleAfterAsyncPrefetch<TransitionEdge::kBegin>(
      std::exception_ptr, LumiTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;
  template std::exception_ptr
  TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::runModuleAfterAsyncPrefetch<TransitionEdge::kEnd>(
      std::exception_ptr, LumiTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;
  template bool TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::runModule<TransitionEdge::kBegin>(
      RunTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);
  template bool TransitionWorker<RunTransitionInfo, TransitionPhaseStream>::runModule<TransitionEdge::kEnd>(
      RunTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);
  template bool TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::runModule<TransitionEdge::kBegin>(
      LumiTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);
  template bool TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>::runModule<TransitionEdge::kEnd>(
      LumiTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);
}  // namespace edm
