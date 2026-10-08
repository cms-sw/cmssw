#ifndef FWCore_Framework_maker_TransitionWorker_ProcessBlock_h
#define FWCore_Framework_maker_TransitionWorker_ProcessBlock_h

/*----------------------------------------------------------------------

Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

A worker will not actually call through to the module unless it is
in a Ready state.  After a module is actually run, the state will not
be Ready.  The Ready state can only be reestablished by doing a reset().

Pre/post module signals are posted only in the Ready state.

Execution statistics are kept here.

If a module has thrown an exception during execution, that exception
will be rethrown if the worker is entered again and the state is not Ready.
In other words, execution results (status) are cached and reused until
the worker is reset().

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/Framework/interface/maker/ModuleSignalSentry.h"
#include "FWCore/Framework/interface/maker/ModuleAttributes.h"
#include "FWCore/Framework/interface/maker/TransitionWorker_Common.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/ProductResolverIndexAndSkipBit.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <array>
#include <atomic>
#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>

namespace edm {
  class EventSetupImpl;
  class ProductResolverIndexAndSkipBit;

  namespace workerhelper {
    template <TransitionEdge E>
    class CallProcessImpl;
  }
  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  struct TransitionPhaseGlobal;

  template <>
  class TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal> : public Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}

    void reset() { resetBase(); }

    template <TransitionEdge E>
    void doWorkAsync(WaitingTaskHolder iTask,
                     ProcessBlockTransitionInfo const& iTransitionInfo,
                     ServiceToken const& iToken,
                     StreamID iStreamID,
                     ParentContext const& iParentContext,
                     GlobalContext const* iContext) noexcept {
      this->doWorkAsyncImpl<E>(std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

  protected:
    //Called by GlobalSchedule::processOneGlobalAsync, UnscheduledCallProducer::runAccumulatorsAsync, WorkerInPath::runWorkerAsync, UnscheduledProductResolver::prefetchAsync_
    template <TransitionEdge E>
    void doWorkAsyncImpl(WaitingTaskHolder,
                         ProcessBlockTransitionInfo const&,
                         ServiceToken const&,
                         StreamID,
                         ParentContext const&,
                         GlobalContext const*) noexcept;

    template <TransitionEdge E>
    friend class workerhelper::CallProcessImpl;

    virtual bool implDoBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) = 0;
    virtual bool implDoEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) = 0;

  private:
    template <TransitionEdge E>
    bool runModule(ProcessBlockTransitionInfo const&, StreamID, ParentContext const&, GlobalContext const*);

    void prefetchAsync(WaitingTaskHolder,
                       ServiceToken const&,
                       ParentContext const&,
                       ProcessBlockTransitionInfo const&,
                       Transition) noexcept;

    void emitPostModuleGlobalPrefetchingSignal() {
      actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(),
                                                       moduleCallingContext_);
    }

    template <TransitionEdge E>
    std::exception_ptr runModuleAfterAsyncPrefetch(std::exception_ptr,
                                                   ProcessBlockTransitionInfo const&,
                                                   StreamID,
                                                   ParentContext const&,
                                                   GlobalContext const*) noexcept;

    template <TransitionEdge E>
    class RunModuleTask : public WaitingTask {
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
  };
  namespace workerhelper {
    template <>
    class CallProcessImpl<TransitionEdge::kBegin> {
    public:
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
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
    };
    template <>
    class CallProcessImpl<TransitionEdge::kEnd> {
    public:
      static bool call(TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>* iWorker,
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
    };
  }  // namespace workerhelper

  inline void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::prefetchAsync(
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
  inline void TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::doWorkAsyncImpl(
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
  inline std::exception_ptr
  TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModuleAfterAsyncPrefetch(
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
  inline bool TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>::runModule(
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

}  // namespace edm
#endif
