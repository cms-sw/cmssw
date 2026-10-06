#ifndef FWCore_Framework_Worker_h
#define FWCore_Framework_Worker_h

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
#include "FWCore/Framework/interface/maker/WorkerParams.h"
#include "FWCore/Framework/interface/maker/ModuleSignalSentry.h"
#include "FWCore/Framework/interface/maker/ModuleAttributes.h"
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
  class EventPrincipal;
  class EventSetupImpl;
  class EarlyDeleteHelper;
  class ProductResolverIndexAndSkipBit;

  namespace eventsetup {
    struct ComponentDescription;
    class ESRecordsToProductResolverIndices;
  }  // namespace eventsetup

  class Worker {
  public:
    enum State { Ready, Pass, Fail, Exception };
    using Types = edm::modules::Type;
    using ConcurrencyTypes = edm::modules::Concurrency;
    struct TaskQueueAdaptor {
      SerialTaskQueueChain* serial_ = nullptr;
      LimitedTaskQueue* limited_ = nullptr;

      TaskQueueAdaptor() = default;
      TaskQueueAdaptor(SerialTaskQueueChain* iChain) : serial_(iChain) {}
      TaskQueueAdaptor(LimitedTaskQueue* iLimited) : limited_(iLimited) {}

      operator bool() { return serial_ != nullptr or limited_ != nullptr; }

      template <class F>
      void push(oneapi::tbb::task_group& iG, F&& iF) {
        if (serial_) {
          serial_->push(iG, iF);
        } else {
          limited_->push(iG, iF);
        }
      }
    };

    Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions);
    virtual ~Worker();

    Worker(Worker const&) = delete;             // Disallow copying and moving
    Worker& operator=(Worker const&) = delete;  // Disallow copying and moving

    void clearModule() {
      moduleValid_ = false;
      doClearModule();
    }

    void callWhenDoneAsync(WaitingTaskHolder task) { waitingTasks_.add(std::move(task)); }

    ModuleDescription const* description() const noexcept {
      if (moduleValid_) {
        return moduleCallingContext_.moduleDescription();
      }
      return nullptr;
    }
    ///The signals are required to live longer than the last call to 'doWork'
    /// this was done to improve performance based on profiling
    void setActivityRegistry(std::shared_ptr<ActivityRegistry> areg);

    virtual Types moduleType() const = 0;
    virtual ConcurrencyTypes moduleConcurrencyType() const = 0;

    //NOTE: calling state() is done to force synchronization across threads
    State state() const noexcept { return state_; }

    virtual bool matchesBaseClassPointer(void const* iPtr) const noexcept = 0;
    // Used in PuttableProductResolver
    edm::WaitingTaskList& waitingTaskList() noexcept { return waitingTasks_; }

  protected:
    void resetBase() {
      cached_exception_ = std::exception_ptr();
      state_ = Ready;
      waitingTasks_.reset();
      workStarted_ = false;
    }

    virtual void doClearModule() = 0;

    void resetModuleDescription(ModuleDescription const*);

    ActivityRegistry* activityRegistry() { return actReg_.get(); }

    virtual void itemsToGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;
    virtual void itemsMayGet(BranchType, std::vector<ProductResolverIndexAndSkipBit>&) const = 0;

    virtual std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType) const = 0;

    virtual std::vector<ESResolverIndex> const& esItemsToGetFrom(Transition) const = 0;
    virtual std::vector<ESRecordIndex> const& esRecordsToGetFrom(Transition) const = 0;

    virtual TaskQueueAdaptor serializeRunModule() = 0;

    bool shouldRethrowException(std::exception_ptr iPtr,
                                ParentContext const& parentContext,
                                bool isEvent,
                                bool isTryToContinue) const noexcept;
    void checkForShouldTryToContinue(ModuleDescription const&);

    bool setPassed() {
      state_ = Pass;
      return true;
    }

    bool setFailed() {
      state_ = Fail;
      return false;
    }

    std::exception_ptr setException(std::exception_ptr iException) {
      cached_exception_ = iException;  // propagate_const<T> has no reset() function
      state_ = Exception;
      return cached_exception_;
    }

    void esPrefetchAsync(WaitingTaskHolder, EventSetupImpl const&, Transition, ServiceToken const&) noexcept;
    void edPrefetchAsync(WaitingTaskHolder, ServiceToken const&, Principal const&) const noexcept;

    bool needsESPrefetching(Transition iTrans) const noexcept {
      return iTrans < edm::Transition::NumberOfEventSetupTransitions ? not esItemsToGetFrom(iTrans).empty() : false;
    }

    void emitPostModuleEventPrefetchingSignal() {
      actReg_->postModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(), moduleCallingContext_);
    }

    void emitPostModuleStreamPrefetchingSignal() {
      actReg_->postModuleStreamPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(),
                                                       moduleCallingContext_);
    }

    void emitPostModuleGlobalPrefetchingSignal() {
      actReg_->postModuleGlobalPrefetchingSignal_.emit(*moduleCallingContext_.getGlobalContext(),
                                                       moduleCallingContext_);
    }

    std::atomic<State> state_;

    ModuleCallingContext moduleCallingContext_;

    ExceptionToActionTable const* actions_;                         // memory assumed to be managed elsewhere
    CMS_THREAD_GUARD(state_) std::exception_ptr cached_exception_;  // if state is 'exception'

    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.

    edm::WaitingTaskList waitingTasks_;
    std::atomic<bool> workStarted_;
    bool moduleValid_ = true;
    bool shouldTryToContinue_ = false;
    bool beginSucceeded_ = false;
  };

}  // namespace edm
#endif
