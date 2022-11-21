
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

namespace edm {
  namespace {
    class ModuleBeginJobSignalSentry {
    public:
      ModuleBeginJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md) : a_(a), md_(&md) {
        if (a_)
          a_->preModuleBeginJobSignal_(*md_);
      }
      ~ModuleBeginJobSignalSentry() {
        if (a_)
          a_->postModuleBeginJobSignal_(*md_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleEndJobSignalSentry {
    public:
      ModuleEndJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md) : a_(a), md_(&md) {
        if (a_)
          a_->preModuleEndJobSignal_(*md_);
      }
      ~ModuleEndJobSignalSentry() {
        if (a_)
          a_->postModuleEndJobSignal_(*md_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleBeginStreamSignalSentry {
    public:
      ModuleBeginStreamSignalSentry(ActivityRegistry* a, StreamContext const& sc, ModuleCallingContext const& mcc)
          : a_(a), sc_(sc), mcc_(mcc) {
        if (a_)
          a_->preModuleBeginStreamSignal_(sc_, mcc_);
      }
      ~ModuleBeginStreamSignalSentry() {
        if (a_)
          a_->postModuleBeginStreamSignal_(sc_, mcc_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
    };

    class ModuleEndStreamSignalSentry {
    public:
      ModuleEndStreamSignalSentry(ActivityRegistry* a, StreamContext const& sc, ModuleCallingContext const& mcc)
          : a_(a), sc_(sc), mcc_(mcc) {
        if (a_)
          a_->preModuleEndStreamSignal_(sc_, mcc_);
      }
      ~ModuleEndStreamSignalSentry() {
        if (a_)
          a_->postModuleEndStreamSignal_(sc_, mcc_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
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
        ranAcquireWithoutException_(false) {}

  Worker::~Worker() {}

  void Worker::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) { actReg_ = areg; }

  bool Worker::shouldRethrowException(std::exception_ptr iPtr, ParentContext const& parentContext, bool isEvent) const {
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

      ModuleCallingContext tempContext(description(), ModuleCallingContext::State::kInvalid, parentContext, nullptr);

      // If we are processing an endpath and the module was scheduled, treat SkipEvent or FailPath
      // as IgnoreCompletely, so any subsequent OutputModules are still run.
      // For unscheduled modules only treat FailPath as IgnoreCompletely but still allow SkipEvent to throw
      ModuleCallingContext const* top_mcc = tempContext.getTopModuleCallingContext();
      if (top_mcc->type() == ParentContext::Type::kPlaceInPath &&
          top_mcc->placeInPathContext()->pathContext()->isEndPath()) {
        if ((action == exception_actions::SkipEvent && tempContext.type() == ParentContext::Type::kPlaceInPath) ||
            action == exception_actions::FailPath) {
          action = exception_actions::IgnoreCompletely;
        }
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
                                         EventPrincipal const* iPrincipal) {
    successTask->increment_ref_count();

    ServiceWeakToken weakToken = token;
    auto choiceTask =
        edm::make_waiting_task([id, successTask, iPrincipal, this, weakToken, &group](std::exception_ptr const*) {
          ServiceRegistry::Operate guard(weakToken.lock());
          // There is no reasonable place to rethrow, and implDoPrePrefetchSelection() should not throw in the first place.
          CMS_SA_ALLOW try {
            if (not implDoPrePrefetchSelection(id, *iPrincipal, &moduleCallingContext_)) {
              timesRun_.fetch_add(1, std::memory_order_relaxed);
              setPassed<true>();
              waitingTasks_.doneWaiting(nullptr);
              //TBB requires that destroyed tasks have count 0
              if (0 == successTask->decrement_ref_count()) {
                TaskSentry s(successTask);
              }
              return;
            }
          } catch (...) {
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
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal->prefetchAsync(
            choiceHolder, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
    choiceHolder.doneWaiting(std::exception_ptr{});
  }

  void Worker::esPrefetchAsync(WaitingTaskHolder iTask,
                               EventSetupImpl const& iImpl,
                               Transition iTrans,
                               ServiceToken const& iToken) {
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

  void Worker::edPrefetchAsync(WaitingTaskHolder iTask, ServiceToken const& token, Principal const& iPrincipal) const {
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

  size_t Worker::transformIndex(edm::BranchDescription const&) const { return -1; }
  void Worker::doTransformAsync(WaitingTaskHolder iTask,
                                size_t iTransformIndex,
                                EventPrincipal const& iPrincipal,
                                ServiceToken const& iToken,
                                StreamID,
                                ModuleCallingContext const& mcc,
                                StreamContext const*) {
    ServiceWeakToken weakToken = iToken;

    //Need to make the services available early so other services can see them
    auto task = make_waiting_task([this, iTask, weakToken, &iPrincipal, iTransformIndex, parent = mcc.parent()](
                                      std::exception_ptr const* iExcept) mutable {
      if (iExcept) {
        iTask.doneWaiting(*iExcept);
        return;
      }
      implDoTransformAsync(iTask, iTransformIndex, iPrincipal, parent, weakToken);
    });

    //NOTE: need different ModuleCallingContext. The ProductResolver will copy the context in order to get
    // a longer lifetime than this function call.
    iPrincipal.prefetchAsync(
        WaitingTaskHolder(*iTask.group(), task), itemToGetForTransform(iTransformIndex), false, iToken, &mcc);
  }

  void Worker::resetModuleDescription(ModuleDescription const* iDesc) {
    ModuleCallingContext temp(iDesc,
                              moduleCallingContext_.state(),
                              moduleCallingContext_.parent(),
                              moduleCallingContext_.previousModuleOnThread());
    moduleCallingContext_ = temp;
  }

  void Worker::beginJob() {
    try {
      convertException::wrap([&]() {
        ModuleBeginJobSignalSentry cpp(actReg_.get(), *description());
        implBeginJob();
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginJob for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::endJob() {
    try {
      convertException::wrap([&]() {
        ModuleDescription const* desc = description();
        assert(desc != nullptr);
        ModuleEndJobSignalSentry cpp(actReg_.get(), *desc);
        implEndJob();
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endJob for module " << description()->moduleName() << "/'" << description()->moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::beginStream(StreamID id, StreamContext& streamContext) {
    try {
      convertException::wrap([&]() {
        streamContext.setTransition(StreamContext::Transition::kBeginStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ParentContext parentContext(&streamContext);
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        moduleCallingContext_.setState(ModuleCallingContext::State::kRunning);
        ModuleBeginStreamSignalSentry beginSentry(actReg_.get(), streamContext, moduleCallingContext_);
        implBeginStream(id);
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginStream for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::endStream(StreamID id, StreamContext& streamContext) {
    try {
      convertException::wrap([&]() {
        streamContext.setTransition(StreamContext::Transition::kEndStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ParentContext parentContext(&streamContext);
        ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
        moduleCallingContext_.setState(ModuleCallingContext::State::kRunning);
        ModuleEndStreamSignalSentry endSentry(actReg_.get(), streamContext, moduleCallingContext_);
        implEndStream(id);
      });
    } catch (cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endStream for module " << description()->moduleName() << "/'" << description()->moduleLabel()
          << "'";
      ex.addContext(ost.str());
      throw;
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
                          WaitingTaskWithArenaHolder& holder) {
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    try {
      convertException::wrap([&]() { this->implDoAcquire(info, &moduleCallingContext_, holder); });
    } catch (cms::Exception& ex) {
      edm::exceptionContext(ex, moduleCallingContext_);
      if (shouldRethrowException(std::current_exception(), parentContext, true)) {
        timesRun_.fetch_add(1, std::memory_order_relaxed);
        throw;
      }
    }
  }

  void Worker::runAcquireAfterAsyncPrefetch(std::exception_ptr iEPtr,
                                            EventTransitionInfo const& eventTransitionInfo,
                                            ParentContext const& parentContext,
                                            WaitingTaskWithArenaHolder holder) {
    ranAcquireWithoutException_ = false;
    std::exception_ptr exceptionPtr;
    if (iEPtr) {
      if (shouldRethrowException(iEPtr, parentContext, true)) {
        exceptionPtr = iEPtr;
      }
      moduleCallingContext_.setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    } else {
      // Caught exception is propagated via WaitingTaskWithArenaHolder
      CMS_SA_ALLOW try {
        runAcquire(eventTransitionInfo, parentContext, holder);
        ranAcquireWithoutException_ = true;
      } catch (...) {
        exceptionPtr = std::current_exception();
      }
    }
    // It is important this is after runAcquire completely finishes
    holder.doneWaiting(exceptionPtr);
  }

  std::exception_ptr Worker::handleExternalWorkException(std::exception_ptr iEPtr, ParentContext const& parentContext) {
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
                                                                           ParentContext const& parentContext)
      : m_worker(worker), m_runModuleTask(runModuleTask), m_group(group), m_parentContext(parentContext) {}

  void Worker::HandleExternalWorkExceptionTask::execute() {
    auto excptr = exceptionPtr();
    WaitingTaskHolder holder(*m_group, m_runModuleTask);
    if (excptr) {
      holder.doneWaiting(m_worker->handleExternalWorkException(excptr, m_parentContext));
    }
  }
}  // namespace edm
