#include "FWCore/Framework/interface/maker/TransitionWorker_ProcessBlock.h"

namespace edm {
  namespace workerhelper {
    template <>
    class CallProcessImpl<TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context);
    };

    template <>
    class CallProcessImpl<TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
                       StreamID,
                       ProcessBlockTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       GlobalContext const* context);
    };
  }  // namespace workerhelper

  template <TransitionEdge E>
  class TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::RunModuleTask : public WaitingTask {
  public:
    RunModuleTask(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* worker,
                  ProcessBlockTransitionInfo const& transitionInfo,
                  ServiceToken const& token,
                  StreamID streamID,
                  ParentContext const& parentContext,
                  GlobalContext const* context,
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
      m_worker->emitPostModuleGlobalPrefetchingSignal();

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
          queue.push(*m_group, std::move(f));
          return;
        }
      }

      m_worker->runModuleAfterAsyncPrefetch<E>(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
    }

  private:
    TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* m_worker;
    ProcessBlockTransitionInfo m_transitionInfo;
    StreamID m_streamID;
    ParentContext const m_parentContext;
    GlobalContext const* m_context;
    ServiceWeakToken m_serviceToken;
    oneapi::tbb::task_group* m_group;
  };

  namespace workerhelper {
    bool CallProcessImpl<TransitionEdge::kBegin>::call(
        TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
        StreamID,
        ProcessBlockTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        GlobalContext const* context) {
      struct SignalTraits {
        using Context = GlobalContext;
        static void preModuleSignal(ActivityRegistry* areg,
                                    GlobalContext const* context,
                                    ModuleCallingContext const* moduleCallingContext) {
          areg->preModuleBeginProcessBlockSignal_.emit(*context, *moduleCallingContext);
        }
        static void postModuleSignal(ActivityRegistry* areg,
                                     GlobalContext const* context,
                                     ModuleCallingContext const* moduleCallingContext) {
          areg->postModuleBeginProcessBlockSignal_.emit(*context, *moduleCallingContext);
        }
      };
      ModuleSignalSentry<SignalTraits> cpp(actReg, context, mcc);
      cpp.preModuleSignal();
      auto returnValue = iWorker->implDoBeginProcessBlock(info.principal(), mcc);
      cpp.postModuleSignal();
      iWorker->beginSucceeded_ = true;
      return returnValue;
    }

    bool CallProcessImpl<TransitionEdge::kEnd>::call(
        TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
        StreamID,
        ProcessBlockTransitionInfo const& info,
        ActivityRegistry* actReg,
        ModuleCallingContext const* mcc,
        GlobalContext const* context) {
      if (iWorker->beginSucceeded_) {
        iWorker->beginSucceeded_ = false;

        struct SignalTraits {
          using Context = GlobalContext;
          static void preModuleSignal(ActivityRegistry* areg,
                                      GlobalContext const* context,
                                      ModuleCallingContext const* moduleCallingContext) {
            areg->preModuleEndProcessBlockSignal_.emit(*context, *moduleCallingContext);
          }
          static void postModuleSignal(ActivityRegistry* areg,
                                       GlobalContext const* context,
                                       ModuleCallingContext const* moduleCallingContext) {
            areg->postModuleEndProcessBlockSignal_.emit(*context, *moduleCallingContext);
          }
        };

        ModuleSignalSentry<SignalTraits> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoEndProcessBlock(info.principal(), mcc);
        cpp.postModuleSignal();
        return returnValue;
      }
      return true;
    }
  }  // namespace workerhelper

  void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::emitPostModuleGlobalPrefetchingSignal() {
    actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);
  }

  void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::prefetchAsync(
      WaitingTaskHolder iTask,
      ServiceToken const& token,
      ParentContext const& parentContext,
      ProcessBlockTransitionInfo const& transitionInfo,
      Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    actReg_->preModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);

    edPrefetchAsync(iTask, token, principal);
  }

  template <TransitionEdge E>
  void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl(
      WaitingTaskHolder task,
      ProcessBlockTransitionInfo const& transitionInfo,
      ServiceToken const& token,
      StreamID streamID,
      ParentContext const& parentContext,
      GlobalContext const* context) noexcept {
    //Need to check workStarted_ before adding to waitingTasks_
    bool expected = false;
    bool workStarted = workStarted_.compare_exchange_strong(expected, true);

    waitingTasks_.add(task);

    if (workStarted) {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

      WaitingTask* moduleTask =
          new RunModuleTask<E>(this, transitionInfo, token, streamID, parentContext, context, task.group());
      auto group = task.group();
      Transition transition;
      if constexpr (E == TransitionEdge::kBegin) {
        transition = Transition::BeginProcessBlock;
      } else {
        transition = Transition::EndProcessBlock;
      }
      prefetchAsync(WaitingTaskHolder(*group, moduleTask), token, parentContext, transitionInfo, transition);
    }
  }

  template <TransitionEdge E>
  std::exception_ptr TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      ProcessBlockTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      GlobalContext const* context) noexcept {
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

  template <TransitionEdge E>
  bool TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModule(
      ProcessBlockTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      GlobalContext const* context) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        rc = workerhelper::CallProcessImpl<E>::call(
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

  //force instantiation of the template methods for the two transitions
  template void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl<
      TransitionEdge::kBegin>(WaitingTaskHolder,
                              ProcessBlockTransitionInfo const&,
                              ServiceToken const&,
                              StreamID,
                              ParentContext const&,
                              GlobalContext const*) noexcept;
  template void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl<
      TransitionEdge::kEnd>(WaitingTaskHolder,
                            ProcessBlockTransitionInfo const&,
                            ServiceToken const&,
                            StreamID,
                            ParentContext const&,
                            GlobalContext const*) noexcept;
  template std::exception_ptr TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::
      runModuleAfterAsyncPrefetch<TransitionEdge::kBegin>(std::exception_ptr,
                                                          ProcessBlockTransitionInfo const&,
                                                          StreamID,
                                                          ParentContext const&,
                                                          GlobalContext const*) noexcept;
  template std::exception_ptr
  TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch<TransitionEdge::kEnd>(
      std::exception_ptr,
      ProcessBlockTransitionInfo const&,
      StreamID,
      ParentContext const&,
      GlobalContext const*) noexcept;
  template bool TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModule<TransitionEdge::kBegin>(
      ProcessBlockTransitionInfo const&, StreamID, ParentContext const&, GlobalContext const*);
  template bool TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModule<TransitionEdge::kEnd>(
      ProcessBlockTransitionInfo const&, StreamID, ParentContext const&, GlobalContext const*);
}  // namespace edm
