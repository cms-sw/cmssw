
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
    timesRun_(),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(Ready),
    moduleCallingContext_(&iMD),
    actions_(iActions),
    cached_exception_(),
    actReg_(),
    earlyDeleteHelper_(nullptr)
  {
  }

  Worker::~Worker() {
  }

  void Worker::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) {
    actReg_ = areg;
  }

  void Worker::prefetchAsync(WaitingTask* iTask, Principal const& iPrincipal) {
    // Prefetch products the module declares it consumes (not including the products it maybe consumes)
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFromEvent();
    
    //Need to be sure the ref count isn't set to 0 immediately
    iTask->increment_ref_count();
    for(auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if(productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask,productResolverIndex, skipCurrentProcess, &moduleCallingContext_);
      }
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

  void Worker::pathFinished(EventPrincipal const& iEvent) {
    if(earlyDeleteHelper_) {
      earlyDeleteHelper_->pathFinished(iEvent);
    }
  }
  void Worker::postDoEvent(EventPrincipal const& iEvent) {
    if(earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }
}
