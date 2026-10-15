#ifndef FWCore_Framework_TransitionWorker_Event_h
#define FWCore_Framework_TransitionWorker_Event_h

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
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionEdge.h"
#include "FWCore/Framework/interface/maker/TransitionWorker_Common.h"
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/Framework/interface/maker/ModuleSignalSentry.h"
#include "FWCore/Framework/interface/maker/ModuleAttributes.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
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
  class EventPrincipal;
  class EventSetupImpl;
  class EarlyDeleteHelper;
  class ProductResolverIndexAndSkipBit;

  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  template <>
  class TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal> : public TransitionWorkerBase {
  public:
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
        : TransitionWorkerBase(iMD, iActions),
          numberOfPathsOn_(0),
          numberOfPathsLeftToRun_(0),
          earlyDeleteHelper_(nullptr),
          ranAcquireWithoutException_(false) {}

    virtual bool hasAccumulator() const noexcept = 0;

    //Run only for OutputModules. Causes TriggerResults information to be used to decide if module should run.
    void prePrefetchSelectionAsync(oneapi::tbb::task_group&,
                                   WaitingTask* task,
                                   ServiceToken const&,
                                   StreamID stream,
                                   EventPrincipal const*) noexcept;

    void prePrefetchSelectionAsync(
        oneapi::tbb::task_group&, WaitingTask* task, ServiceToken const&, StreamID stream, void const*) noexcept {
      assert(false);
    }

    //Called by Path to inject PathStatus and StreamSchedule to inject TriggerResults and PathStatus for empty paths.
    std::exception_ptr runModuleDirectly(EventTransitionInfo const&,
                                         StreamID,
                                         ParentContext const&,
                                         StreamContext const*) noexcept;

    //called by TransformingProductResolver so only for global Event
    virtual size_t transformIndex(edm::ProductDescription const&) const noexcept = 0;
    void doTransformAsync(WaitingTaskHolder,
                          size_t iTransformIndex,
                          EventPrincipal const&,
                          ServiceToken const&,
                          StreamID,
                          ModuleCallingContext const&,
                          StreamContext const*) noexcept;

    // Called if filter earlier in the path has failed.
    void skipOnPath(EventPrincipal const& iEvent);

    void reset() {
      resetBase();
      numberOfPathsLeftToRun_ = numberOfPathsOn_;
    }

    void postDoEvent(EventPrincipal const&);

    void setEarlyDeleteHelper(EarlyDeleteHelper* iHelper);

    //Only for global Event
    void addedToPath() noexcept { ++numberOfPathsOn_; }

    template <TransitionEdge E>
    void doWorkAsync(WaitingTaskHolder iTask,
                     EventTransitionInfo const& iTransitionInfo,
                     ServiceToken const& iToken,
                     StreamID iStreamID,
                     ParentContext const& iParentContext,
                     StreamContext const* iContext) noexcept {
      if constexpr (E == TransitionEdge::kBegin) {
        this->doWorkAsyncImpl(std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
      }
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoEDPrefetchingAsync(WaitingTaskHolder iTask,
                                    EventTransitionInfo const& iTransitionInfo,
                                    ServiceToken const& iToken,
                                    StreamID iStreamID,
                                    ParentContext const& iParentContext,
                                    StreamContext const* iContext) noexcept {
      if constexpr (E == TransitionEdge::kBegin) {
        this->doWorkNoEDPrefetchingAsyncImpl(
            std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
      }
    }

  private:
    //Called by GlobalSchedule::processOneGlobalAsync, UnscheduledCallProducer::runAccumulatorsAsync, WorkerInPath::runWorkerAsync, UnscheduledProductResolver::prefetchAsync_
    void doWorkAsyncImpl(WaitingTaskHolder,
                         EventTransitionInfo const&,
                         ServiceToken const&,
                         StreamID,
                         ParentContext const&,
                         StreamContext const*) noexcept;

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    void doWorkNoEDPrefetchingAsyncImpl(WaitingTaskHolder,
                                        EventTransitionInfo const&,
                                        ServiceToken const&,
                                        StreamID,
                                        ParentContext const&,
                                        StreamContext const*) noexcept;

  protected:
    virtual bool implDo(EventTransitionInfo const&, ModuleCallingContext const*) = 0;

    virtual void itemsToGetForSelection(std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual bool implNeedToRunSelection() const noexcept = 0;

    virtual void implDoAcquire(EventTransitionInfo const&, ModuleCallingContext const*, WaitingTaskHolder&&) = 0;

    virtual void implDoTransformAsync(WaitingTaskHolder,
                                      size_t iTransformIndex,
                                      EventPrincipal const&,
                                      ParentContext const&,
                                      ServiceWeakToken const&) noexcept = 0;
    virtual ProductResolverIndex itemToGetForTransform(size_t iTransformIndex) const noexcept = 0;

    virtual bool implDoPrePrefetchSelection(StreamID, EventPrincipal const&, ModuleCallingContext const*) = 0;

  private:
    bool call(StreamID,
              EventTransitionInfo const& info,
              ActivityRegistry* actReg,
              ModuleCallingContext* mcc,
              StreamContext const* context);
    bool needToRunSelection() const noexcept { return this->implNeedToRunSelection(); }

    bool runModule(EventTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*);

    //Only used by Event and only for OutputModule
    virtual void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                              ModuleCallingContext const& moduleCallingContext,
                                              Principal const& iPrincipal) const noexcept = 0;

    void prefetchAsync(
        WaitingTaskHolder, ServiceToken const&, ParentContext const&, EventTransitionInfo const&, Transition) noexcept;

    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    virtual bool hasAcquire() const noexcept = 0;

    std::exception_ptr runModuleAfterAsyncPrefetch(
        std::exception_ptr, EventTransitionInfo const&, StreamID, ParentContext const&, StreamContext const*) noexcept;

    // runAcquire() must take a copy of WaitingTaskHolder
    // see comment in runAcquireAfterAsyncPrefetch() definition
    void runAcquire(EventTransitionInfo const&, ParentContext const&, WaitingTaskHolder);

    //Only for Event
    void runAcquireAfterAsyncPrefetch(std::exception_ptr,
                                      EventTransitionInfo const&,
                                      ParentContext const&,
                                      WaitingTaskHolder) noexcept;

    std::exception_ptr handleExternalWorkException(std::exception_ptr iEPtr,
                                                   ParentContext const& parentContext) noexcept;

    class RunModuleTask;
    class AcquireTask;

    // This class does nothing unless there is an exception originating
    // in an "External Worker". In that case, it handles converting the
    // exception to a CMS exception and adding context to the exception.
    class HandleExternalWorkExceptionTask;

    int numberOfPathsOn_;
    std::atomic<int> numberOfPathsLeftToRun_;

    edm::propagate_const<EarlyDeleteHelper*> earlyDeleteHelper_;

    bool ranAcquireWithoutException_;
  };
}  // namespace edm
#endif
