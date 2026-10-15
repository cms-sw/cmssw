#include "FWCore/Framework/interface/maker/TransitionWorker_InputProcessBlock.h"

namespace edm {
  class TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::RunModuleTask : public WaitingTask {
  public:
    RunModuleTask(TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>* worker,
                  InputProcessBlockTransitionInfo const& transitionInfo,
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
            worker->runModuleAfterAsyncPrefetch(ptr, info, streamID, parentContext, sContext);
          };
          queue.push(*m_group, std::move(f));
          return;
        }
      }

      m_worker->runModuleAfterAsyncPrefetch(excptr, m_transitionInfo, m_streamID, m_parentContext, m_context);
    }

  private:
    TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>* m_worker;
    InputProcessBlockTransitionInfo m_transitionInfo;
    StreamID m_streamID;
    ParentContext const m_parentContext;
    GlobalContext const* m_context;
    ServiceWeakToken m_serviceToken;
    oneapi::tbb::task_group* m_group;
  };

  void TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::emitPostModuleGlobalPrefetchingSignal() {
    actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);
  }

  void TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::prefetchAsync(
      WaitingTaskHolder iTask,
      ServiceToken const& token,
      ParentContext const& parentContext,
      InputProcessBlockTransitionInfo const& transitionInfo,
      Transition iTransition) noexcept {
    Principal const& principal = transitionInfo.principal();

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching, parentContext, nullptr);

    actReg_->preModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(), moduleCallingContext_);

    edPrefetchAsync(iTask, token, principal);
  }

  void TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl(
      WaitingTaskHolder task,
      InputProcessBlockTransitionInfo const& transitionInfo,
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
          new RunModuleTask(this, transitionInfo, token, streamID, parentContext, context, task.group());
      auto group = task.group();
      prefetchAsync(WaitingTaskHolder(*group, moduleTask),
                    token,
                    parentContext,
                    transitionInfo,
                    Transition::AccessInputProcessBlock);
    }
  }

  std::exception_ptr
  TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch(
      std::exception_ptr iEPtr,
      InputProcessBlockTransitionInfo const& transitionInfo,
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
      CMS_SA_ALLOW try { runModule(transitionInfo, streamID, parentContext, context); } catch (...) {
        exceptionPtr = std::current_exception();
      }
    } else {
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
    waitingTasks_.doneWaiting(exceptionPtr);
    return exceptionPtr;
  }

  bool TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModule(
      InputProcessBlockTransitionInfo const& transitionInfo,
      StreamID streamID,
      ParentContext const& parentContext,
      GlobalContext const* context) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);

    bool rc = true;
    try {
      convertException::wrap([&]() {
        {
          struct SignalTraits {
            using Context = GlobalContext;
            static void preModuleSignal(ActivityRegistry* areg,
                                        GlobalContext const* context,
                                        ModuleCallingContext const* moduleCallingContext) {
              areg->preModuleAccessInputProcessBlockSignal_.emit(*context, *moduleCallingContext);
            }
            static void postModuleSignal(ActivityRegistry* areg,
                                         GlobalContext const* context,
                                         ModuleCallingContext const* moduleCallingContext) {
              areg->postModuleAccessInputProcessBlockSignal_.emit(*context, *moduleCallingContext);
            }
          };
          ModuleSignalSentry<SignalTraits> cpp(actReg_.get(), context, &moduleCallingContext_);
          cpp.preModuleSignal();
          rc = this->implDoAccessInputProcessBlock(transitionInfo.principal(), &moduleCallingContext_);
          cpp.postModuleSignal();
        }

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
}  // namespace edm
