
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Concurrency/interface/include_first_syncWait.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/EarlyDeleteHelper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {
  namespace {
    class ModuleBeginJobTraits {
    public:
      using Context = GlobalContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  GlobalContext const*,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleBeginJobSignal_(*moduleCallingContext->moduleDescription());
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   GlobalContext const*,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleBeginJobSignal_(*moduleCallingContext->moduleDescription());
      }
    };

    class ModuleEndJobTraits {
    public:
      using Context = GlobalContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  GlobalContext const*,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleEndJobSignal_(*moduleCallingContext->moduleDescription());
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   GlobalContext const*,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleEndJobSignal_(*moduleCallingContext->moduleDescription());
      }
    };

    class ModuleBeginStreamTraits {
    public:
      using Context = StreamContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  StreamContext const* streamContext,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleBeginStreamSignal_(*streamContext, *moduleCallingContext);
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   StreamContext const* streamContext,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleBeginStreamSignal_(*streamContext, *moduleCallingContext);
      }
    };

    class ModuleEndStreamTraits {
    public:
      using Context = StreamContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  StreamContext const* streamContext,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleEndStreamSignal_(*streamContext, *moduleCallingContext);
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   StreamContext const* streamContext,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleEndStreamSignal_(*streamContext, *moduleCallingContext);
      }
    };

  }  // namespace

  Worker::Worker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
      : timesRun_(0),
        timesVisited_(0),
        timesPassed_(0),
        timesFailed_(0),
        timesExcept_(0),
        state_(Ready),
        numberOfPathsOn_(0),
        numberOfPathsLeftToRun_(0),
        moduleCallingContext_(&iMD),
        actions_(iActions),
        cached_exception_(),
        actReg_(),
        earlyDeleteHelper_(nullptr),
        workStarted_(false),
        ranAcquireWithoutException_(false) {
    checkForShouldTryToContinue(iMD);
  }

  Worker::~Worker() {}

  void Worker::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) { actReg_ = areg; }

  void Worker::checkForShouldTryToContinue(ModuleDescription const& iDesc) {
    auto pset = edm::pset::Registry::instance()->getMapped(iDesc.parameterSetID());
    if (pset and pset->exists("@shouldTryToContinue")) {
      shouldTryToContinue_ = true;
    }
  }

  bool Worker::shouldRethrowException(std::exception_ptr iPtr,
                                      ParentContext const& parentContext,
                                      bool isEvent,
                                      bool shouldTryToContinue) const noexcept {
    // NOTE: the warning printed as a result of ignoring or failing
    // a module will only be printed during the full true processing
    // pass of this module

    // Get the action corresponding to this exception.  However, if processing
    // something other than an event (e.g. run, lumi) always rethrow.
    if (not isEvent) {
      return true;
    }
    try {
      convertException::wrap([&]() { std::rethrow_exception(iPtr); });
    } catch (cms::Exception& ex) {
      exception_actions::ActionCodes action = actions_->find(ex.category());

      if (action == exception_actions::Rethrow) {
        return true;
      }
      if (action == exception_actions::TryToContinue) {
        if (shouldTryToContinue) {
          edm::printCmsExceptionWarning("TryToContinue", ex);
        }
        return not shouldTryToContinue;
      }
      if (action == exception_actions::IgnoreCompletely) {
        edm::printCmsExceptionWarning("IgnoreCompletely", ex);
        return false;
      }
    }
    return true;
  }

  void Worker::prePrefetchSelectionAsync(oneapi::tbb::task_group& group,
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
                timesRun_.fetch_add(1, std::memory_order_relaxed);
                setPassed<true>();
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
            setException<true>(std::current_exception());
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
      bool skipCurrentProcess = item.skipCurrentProcess();
      if (productResolverIndex != ProductResolverIndexAmbiguous and
          productResolverIndex != ProductResolverIndexInvalid) {
        iPrincipal->prefetchAsync(
            choiceHolder, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
    choiceHolder.doneWaiting(std::exception_ptr{});
  }

  void Worker::esPrefetchAsync(WaitingTaskHolder iTask,
                               EventSetupImpl const& iImpl,
                               Transition iTrans,
                               ServiceToken const& iToken) noexcept {
    if (iTrans >= edm::Transition::NumberOfEventSetupTransitions) {
      return;
    }
    auto const& recs = esRecordsToGetFrom(iTrans);
    auto const& items = esItemsToGetFrom(iTrans);

    assert(items.size() == recs.size());
    if (items.empty()) {
      return;
    }

    for (size_t i = 0; i != items.size(); ++i) {
      if (recs[i] != ESRecordIndex{}) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          rec->prefetchAsync(iTask, items[i], &iImpl, iToken, ESParentContext(&moduleCallingContext_));
        }
      }
    }
  }

  void Worker::edPrefetchAsync(WaitingTaskHolder iTask,
                               ServiceToken const& token,
                               Principal const& iPrincipal) const noexcept {
    // Prefetch products the module declares it consumes
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFrom(iPrincipal.branchType());

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
  }

  void Worker::setEarlyDeleteHelper(EarlyDeleteHelper* iHelper) { earlyDeleteHelper_ = iHelper; }

  size_t Worker::transformIndex(edm::BranchDescription const&) const noexcept { return -1; }
  void Worker::doTransformAsync(WaitingTaskHolder iTask,
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
        WaitingTaskHolder(*iTask.group(), task), itemToGetForTransform(iTransformIndex), false, iToken, &mcc);
  }

  void Worker::resetModuleDescription(ModuleDescription const* iDesc) {
    ModuleCallingContext temp(iDesc,
                              0,
                              moduleCallingContext_.state(),
                              moduleCallingContext_.parent(),
                              moduleCallingContext_.previousModuleOnThread());
    moduleCallingContext_ = temp;
    assert(iDesc);
    checkForShouldTryToContinue(*iDesc);
  }

  void Worker::beginJob(GlobalContext const& globalContext) {
    ParentContext parentContext(&globalContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    ModuleSignalSentry<ModuleBeginJobTraits> sentry(activityRegistry(), &globalContext, &moduleCallingContext_);

    try {
      convertException::wrap([this, &sentry]() {
        beginSucceeded_ = false;
        sentry.preModuleSignal();
        implBeginJob();
        sentry.postModuleSignal();
        beginSucceeded_ = true;
      });
    } catch (cms::Exception& ex) {
      exceptionContext(ex, moduleCallingContext_);
      throw;
    }
  }

  void Worker::endJob(GlobalContext const& globalContext) {
    if (beginSucceeded_) {
      beginSucceeded_ = false;

      ParentContext parentContext(&globalContext);
      ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
      ModuleSignalSentry<ModuleEndJobTraits> sentry(activityRegistry(), &globalContext, &moduleCallingContext_);

      try {
        convertException::wrap([this, &sentry]() {
          sentry.preModuleSignal();
          implEndJob();
          sentry.postModuleSignal();
        });
      } catch (cms::Exception& ex) {
        exceptionContext(ex, moduleCallingContext_);
        throw;
      }
    }
  }

  void Worker::beginStream(StreamID streamID, StreamContext const& streamContext) {
    ParentContext parentContext(&streamContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    ModuleSignalSentry<ModuleBeginStreamTraits> sentry(activityRegistry(), &streamContext, &moduleCallingContext_);

    try {
      convertException::wrap([this, &sentry, streamID]() {
        beginSucceeded_ = false;
        sentry.preModuleSignal();
        implBeginStream(streamID);
        sentry.postModuleSignal();
        beginSucceeded_ = true;
      });
    } catch (cms::Exception& ex) {
      exceptionContext(ex, moduleCallingContext_);
      throw;
    }
  }

  void Worker::endStream(StreamID id, StreamContext const& streamContext) {
    if (beginSucceeded_) {
      beginSucceeded_ = false;

      ParentContext parentContext(&streamContext);
      ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
      ModuleSignalSentry<ModuleEndStreamTraits> sentry(activityRegistry(), &streamContext, &moduleCallingContext_);

      try {
        convertException::wrap([this, &sentry, id]() {
          sentry.preModuleSignal();
          implEndStream(id);
          sentry.postModuleSignal();
        });
      } catch (cms::Exception& ex) {
        exceptionContext(ex, moduleCallingContext_);
        throw;
      }
    }
  }

  void Worker::registerThinnedAssociations(ProductRegistry const& registry, ThinnedAssociationsHelper& helper) {
    try {
      implRegisterThinnedAssociations(registry, helper);
    } catch (cms::Exception& ex) {
      ex.addContext("Calling registerThinnedAssociations() for module " + description()->moduleLabel());
      throw ex;
    }
  }

  void Worker::skipOnPath(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->pathFinished(iEvent);
    }
    if (0 == --numberOfPathsLeftToRun_) {
      waitingTasks_.doneWaiting(cached_exception_);
    }
  }

  void Worker::postDoEvent(EventPrincipal const& iEvent) {
    if (earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }

  void Worker::runAcquire(EventTransitionInfo const& info,
                          ParentContext const& parentContext,
                          WaitingTaskHolder holder) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    try {
      convertException::wrap([&]() { this->implDoAcquire(info, &moduleCallingContext_, std::move(holder)); });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, true, shouldTryToContinue_)) {
        timesRun_.fetch_add(1, std::memory_order_relaxed);
        throw;
      }
    }
  }

  void Worker::runAcquireAfterAsyncPrefetch(std::exception_ptr iEPtr,
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

  std::exception_ptr Worker::handleExternalWorkException(std::exception_ptr iEPtr,
                                                         ParentContext const& parentContext) noexcept {
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

  Worker::HandleExternalWorkExceptionTask::HandleExternalWorkExceptionTask(Worker* worker,
                                                                           oneapi::tbb::task_group* group,
                                                                           WaitingTask* runModuleTask,
                                                                           ParentContext const& parentContext) noexcept
      : m_worker(worker), m_runModuleTask(runModuleTask), m_group(group), m_parentContext(parentContext) {}

  void Worker::HandleExternalWorkExceptionTask::execute() {
    auto excptr = exceptionPtr();
    WaitingTaskHolder holder(*m_group, m_runModuleTask);
    if (excptr) {
      holder.doneWaiting(m_worker->handleExternalWorkException(excptr, m_parentContext));
    }
  }
}  // namespace edm
