#include "FWCore/Framework/interface/maker/TransitionWorker_Event.h"
#include "FWCore/Framework/interface/EarlyDeleteHelper.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"

namespace edm {
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
}  // namespace edm
