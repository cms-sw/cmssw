
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"

namespace edm {
  namespace {
    class ModuleBeginJobSignalSentry {
public:
      ModuleBeginJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md):a_(a), md_(&md) {
        if(a_) a_->preModuleBeginJobSignal_(*md_);
      }
      ~ModuleBeginJobSignalSentry() {
        if(a_) a_->postModuleBeginJobSignal_(*md_);
      }
private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleEndJobSignalSentry {
public:
      ModuleEndJobSignalSentry(ActivityRegistry* a, ModuleDescription const& md):a_(a), md_(&md) {
        if(a_) a_->preModuleEndJobSignal_(*md_);
      }
      ~ModuleEndJobSignalSentry() {
        if(a_) a_->postModuleEndJobSignal_(*md_);
      }
private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      ModuleDescription const* md_;
    };

    class ModuleBeginStreamSignalSentry {
    public:
      ModuleBeginStreamSignalSentry(ActivityRegistry* a,
                                    StreamContext const& sc,
                                    ModuleCallingContext const& mcc) : a_(a), sc_(sc), mcc_(mcc) {
        if(a_) a_->preModuleBeginStreamSignal_(sc_, mcc_);
      }
      ~ModuleBeginStreamSignalSentry() {
        if(a_) a_->postModuleBeginStreamSignal_(sc_, mcc_);
      }
    private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
    };

    class ModuleEndStreamSignalSentry {
    public:
      ModuleEndStreamSignalSentry(ActivityRegistry* a,
                                  StreamContext const& sc,
                                  ModuleCallingContext const& mcc) : a_(a), sc_(sc), mcc_(mcc) {
        if(a_) a_->preModuleEndStreamSignal_(sc_, mcc_);
      }
      ~ModuleEndStreamSignalSentry() {
        if(a_) a_->postModuleEndStreamSignal_(sc_, mcc_);
      }
    private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      StreamContext const& sc_;
      ModuleCallingContext const& mcc_;
    };

  }

  Worker::Worker(ModuleDescription const& iMD, 
		 ExceptionToActionTable const* iActions) :
    timesRun_(0),
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
    workStarted_(false)
  {
  }

  Worker::~Worker() {
  }

  void Worker::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) {
    actReg_ = areg;
  }

  
  void Worker::exceptionContext(
                        cms::Exception& ex,
                        ModuleCallingContext const* mcc) {
    
    ModuleCallingContext const* imcc = mcc;
    while( (imcc->type() == ParentContext::Type::kModule) or
          (imcc->type()  == ParentContext::Type::kInternal) ) {
      std::ostringstream iost;
      if( imcc->state() == ModuleCallingContext::State::kPrefetching ) {
        iost << "Prefetching for module ";
      } else {
        iost << "Calling method for module ";
      }
      iost << imcc->moduleDescription()->moduleName() << "/'"
      << imcc->moduleDescription()->moduleLabel() << "'";

      if(imcc->type() == ParentContext::Type::kInternal) {
        iost << " (probably inside some kind of mixing module)";
        imcc = imcc->internalContext()->moduleCallingContext();
      } else {
        imcc = imcc->moduleCallingContext();
      }
      ex.addContext(iost.str());
    }
    std::ostringstream ost;
    if( imcc->state() == ModuleCallingContext::State::kPrefetching ) {
      ost << "Prefetching for module ";
    } else {
      ost << "Calling method for module ";
    }
    ost << imcc->moduleDescription()->moduleName() << "/'"
    << imcc->moduleDescription()->moduleLabel() << "'";
    ex.addContext(ost.str());
    
    if (imcc->type() == ParentContext::Type::kPlaceInPath) {
      ost.str("");
      ost << "Running path '";
      ost << imcc->placeInPathContext()->pathContext()->pathName() << "'";
      ex.addContext(ost.str());
      auto streamContext =imcc->placeInPathContext()->pathContext()->streamContext();
      if(streamContext) {
        ost.str("");
        edm::exceptionContext(ost,*streamContext);
        ex.addContext(ost.str());
      }
    } else {
      if (imcc->type() == ParentContext::Type::kStream) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->streamContext()) );
        ex.addContext(ost.str());
      } else if(imcc->type() == ParentContext::Type::kGlobal) {
        ost.str("");
        edm::exceptionContext(ost, *(imcc->globalContext()) );
        ex.addContext(ost.str());
      }
    }
  }

  bool Worker::shouldRethrowException(std::exception_ptr iPtr,
                                      ParentContext const& parentContext,
                                      bool isEvent,
                                      TransitionIDValueBase const& iID) const {
    
    // NOTE: the warning printed as a result of ignoring or failing
    // a module will only be printed during the full true processing
    // pass of this module
    
    // Get the action corresponding to this exception.  However, if processing
    // something other than an event (e.g. run, lumi) always rethrow.
    if(not isEvent) {
      return true;
    }
    try {
      convertException::wrap([&]() {
        std::rethrow_exception(iPtr);
      });
    } catch(cms::Exception &ex) {
      exception_actions::ActionCodes action = actions_->find(ex.category());
    
      if(action == exception_actions::Rethrow) {
        return true;
      }
    
      ModuleCallingContext tempContext(&description(),ModuleCallingContext::State::kInvalid, parentContext, nullptr);
      
      // If we are processing an endpath and the module was scheduled, treat SkipEvent or FailPath
      // as IgnoreCompletely, so any subsequent OutputModules are still run.
      // For unscheduled modules only treat FailPath as IgnoreCompletely but still allow SkipEvent to throw
      ModuleCallingContext const* top_mcc = tempContext.getTopModuleCallingContext();
      if(top_mcc->type() == ParentContext::Type::kPlaceInPath &&
         top_mcc->placeInPathContext()->pathContext()->isEndPath()) {
        
        if ((action == exception_actions::SkipEvent && tempContext.type() == ParentContext::Type::kPlaceInPath) ||
            action == exception_actions::FailPath) {
          action = exception_actions::IgnoreCompletely;
        }
      }
      if(action == exception_actions::IgnoreCompletely) {
        edm::printCmsExceptionWarning("IgnoreCompletely", ex);
        return false;
      }
    }
    return true;
  }

  
  void Worker::prefetchAsync(WaitingTask* iTask, ParentContext const& parentContext, Principal const& iPrincipal) {
    // Prefetch products the module declares it consumes (not including the products it maybe consumes)
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFrom(iPrincipal.branchType());

    moduleCallingContext_.setContext(ModuleCallingContext::State::kPrefetching,parentContext,nullptr);
    
    if(iPrincipal.branchType()==InEvent) {
      actReg_->preModuleEventPrefetchingSignal_.emit(*moduleCallingContext_.getStreamContext(),moduleCallingContext_);
    }

    //Need to be sure the ref count isn't set to 0 immediately
    iTask->increment_ref_count();
    for(auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if(productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask,productResolverIndex, skipCurrentProcess, &moduleCallingContext_);
      }
    }
    
    if(iPrincipal.branchType()==InEvent) {
      preActionBeforeRunEventAsync(iTask,moduleCallingContext_,iPrincipal);
    }
    
    if(0 == iTask->decrement_ref_count()) {
      //if everything finishes before we leave this routine, we need to launch the task
      tbb::task::spawn(*iTask);
    }
  }
  
  void Worker::setEarlyDeleteHelper(EarlyDeleteHelper* iHelper) {
    earlyDeleteHelper_=iHelper;
  }
  
  void Worker::resetModuleDescription(ModuleDescription const* iDesc) {
    ModuleCallingContext temp(iDesc,moduleCallingContext_.state(),moduleCallingContext_.parent(),
                              moduleCallingContext_.previousModuleOnThread());
    moduleCallingContext_ = temp;
  }

  void Worker::beginJob() {
    try {
      convertException::wrap([&]() {
        ModuleBeginJobSignalSentry cpp(actReg_.get(), description());
        implBeginJob();
      });
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginJob for module " << description().moduleName() << "/'" << description().moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }
  
  void Worker::endJob() {
    try {
      convertException::wrap([&]() {
        ModuleEndJobSignalSentry cpp(actReg_.get(), description());
        implEndJob();
      });
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endJob for module " << description().moduleName() << "/'" << description().moduleLabel() << "'";
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
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginStream for module " << description().moduleName() << "/'" << description().moduleLabel() << "'";
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
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endStream for module " << description().moduleName() << "/'" << description().moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }
  
  void Worker::skipOnPath() {
    if( 0 == --numberOfPathsLeftToRun_) {
      waitingTasks_.doneWaiting(cached_exception_);
    }
  }

  void Worker::postDoEvent(EventPrincipal const& iEvent) {
    if(earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }
}
