#ifndef FWCore_Framework_maker_TransitionWorker_Stream_h
#define FWCore_Framework_maker_TransitionWorker_Stream_h

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
#include "FWCore/Framework/interface/OccurrenceTraits.h"
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
    template <typename TI, TransitionEdge E>
    class CallStreamImpl;
  }
  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  class RunTransitionInfo;
  class LumiTransitionInfo;
  struct TransitionPhaseGlobal;
  struct TransitionPhaseStream;

  template <typename TI>
  class TransitionWorker<TI, TransitionPhaseStream> : public Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}

    void reset() { resetBase(); }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoEDPrefetchingAsync(WaitingTaskHolder iTask,
                                    TI const& iTransitionInfo,
                                    ServiceToken const& iToken,
                                    StreamID iStreamID,
                                    ParentContext const& iParentContext,
                                    StreamContext const* iContext) noexcept {
      this->template doWorkNoEDPrefetchingAsyncImpl<E>(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

  protected:
    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoEDPrefetchingAsyncImpl(WaitingTaskHolder,
                                        TI const&,
                                        ServiceToken const&,
                                        StreamID,
                                        ParentContext const&,
                                        StreamContext const*) noexcept;

    template <typename O>
    friend class workerhelper::CallStreamImpl;

    virtual bool implDoStreamBegin(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, RunTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamBegin(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;
    virtual bool implDoStreamEnd(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*) = 0;

  private:
    template <TransitionEdge E>
    bool runModule(TI const&, StreamID, ParentContext const&, StreamContext const*);

    void emitPostModuleStreamPrefetchingSignal() {
      actReg_->postModuleStreamPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(),
                                                       moduleCallingContext_);
    }

    template <TransitionEdge E>
    std::exception_ptr runModuleAfterAsyncPrefetch(
        std::exception_ptr, TI const&, StreamID, ParentContext const&, StreamContext const*) noexcept;

    template <TransitionEdge E>
    class RunModuleTask : public WaitingTask {
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
  };
  namespace workerhelper {
    template <>
    class CallStreamImpl<RunTransitionInfo, TransitionEdge::kBegin> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionStreamBegin> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       Arg::Context const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return true;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T*) noexcept {
        return nullptr;
      }
    };
    template <>
    class CallStreamImpl<RunTransitionInfo, TransitionEdge::kEnd> {
    public:
      typedef OccurrenceTraits<RunPrincipal, TransitionActionStreamEnd> Arg;
      static bool call(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       RunTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       StreamContext const* context) {
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
          cpp.postModuleSignal();
          return returnValue;
        }
        return true;
      }
      static void esPrefetchAsync(TransitionWorker<RunTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  RunTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
    };

    template <>
    class CallStreamImpl<LumiTransitionInfo, TransitionEdge::kBegin> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamBegin>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       StreamContext const* context) {
        ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
        cpp.preModuleSignal();
        auto returnValue = iWorker->implDoStreamBegin(id, info, mcc);
        cpp.postModuleSignal();
        iWorker->beginSucceeded_ = true;
        return returnValue;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
    };
    template <>
    class CallStreamImpl<LumiTransitionInfo, TransitionEdge::kEnd> {
    public:
      using Arg = OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamEnd>;
      static bool call(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* iWorker,
                       StreamID id,
                       LumiTransitionInfo const& info,
                       ActivityRegistry* actReg,
                       ModuleCallingContext const* mcc,
                       StreamContext const* context) {
        if (iWorker->beginSucceeded_) {
          iWorker->beginSucceeded_ = false;

          ModuleSignalSentry<Arg> cpp(actReg, context, mcc);
          cpp.preModuleSignal();
          auto returnValue = iWorker->implDoStreamEnd(id, info, mcc);
          cpp.postModuleSignal();
          return returnValue;
        }
        return true;
      }
      static void esPrefetchAsync(TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>* worker,
                                  WaitingTaskHolder waitingTask,
                                  ServiceToken const& token,
                                  LumiTransitionInfo const& info,
                                  Transition transition) noexcept {
        worker->esPrefetchAsync(waitingTask, info.eventSetupImpl(), transition, token);
      }
      template <typename T>
      static bool wantsTransition(T const* iWorker) noexcept {
        return true;
      }
      template <typename T>
      static SerialTaskQueue* pauseGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
      template <typename T>
      static SerialTaskQueue* enableGlobalQueue(T* iWorker) noexcept {
        return nullptr;
      }
    };
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

}  // namespace edm
#endif
