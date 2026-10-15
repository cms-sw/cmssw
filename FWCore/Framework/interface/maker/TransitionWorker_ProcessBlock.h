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
#include "FWCore/Framework/interface/maker/TransitionWorkerBase.h"
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
  class TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal> : public TransitionWorkerBase {
  public:
    enum State { Ready, Pass, Fail, Exception };
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
        : TransitionWorkerBase(iMD, iActions) {}

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

    void emitPostModuleGlobalPrefetchingSignal();

    template <TransitionEdge E>
    std::exception_ptr runModuleAfterAsyncPrefetch(std::exception_ptr,
                                                   ProcessBlockTransitionInfo const&,
                                                   StreamID,
                                                   ParentContext const&,
                                                   GlobalContext const*) noexcept;

    template <TransitionEdge E>
    class RunModuleTask;
  };
}  // namespace edm
#endif
