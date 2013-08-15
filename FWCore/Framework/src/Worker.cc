
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

namespace edm {
  namespace {
    class ModuleBeginJobSignalSentry {
public:
      ModuleBeginJobSignalSentry(ActivityRegistry* a, ModuleDescription& md):a_(a), md_(&md) {
        if(a_) a_->preModuleBeginJobSignal_(*md_);
      }
      ~ModuleBeginJobSignalSentry() {
        if(a_) a_->postModuleBeginJobSignal_(*md_);
      }
private:
      ActivityRegistry* a_;
      ModuleDescription* md_;
    };

    class ModuleEndJobSignalSentry {
public:
      ModuleEndJobSignalSentry(ActivityRegistry* a, ModuleDescription& md):a_(a), md_(&md) {
        if(a_) a_->preModuleEndJobSignal_(*md_);
      }
      ~ModuleEndJobSignalSentry() {
        if(a_) a_->postModuleEndJobSignal_(*md_);
      }
private:
      ActivityRegistry* a_;
      ModuleDescription* md_;
    };

    class ModuleBeginStreamSignalSentry {
    public:
      ModuleBeginStreamSignalSentry(ActivityRegistry* a,
                                    StreamContext const& sc,
                                    ModuleDescription const& md) : a_(a), sc_(sc), md_(md) {
        if(a_) a_->preModuleBeginStreamSignal_(sc_, md_);
      }
      ~ModuleBeginStreamSignalSentry() {
        if(a_) a_->postModuleBeginStreamSignal_(sc_, md_);
      }
    private:
      ActivityRegistry* a_;
      StreamContext const& sc_;
      ModuleDescription const& md_;
    };

    class ModuleEndStreamSignalSentry {
    public:
      ModuleEndStreamSignalSentry(ActivityRegistry* a,
                                  StreamContext const& sc,
                                  ModuleDescription const& md) : a_(a), sc_(sc), md_(md) {
        if(a_) a_->preModuleEndStreamSignal_(sc_, md_);
      }
      ~ModuleEndStreamSignalSentry() {
        if(a_) a_->postModuleEndStreamSignal_(sc_, md_);
      }
    private:
      ActivityRegistry* a_;
      StreamContext const& sc_;
      ModuleDescription const& md_;
    };

    cms::Exception& exceptionContext(ModuleDescription const& iMD,
                                     cms::Exception& iEx) {
      iEx << iMD.moduleName() << "/" << iMD.moduleLabel() << "\n";
      return iEx;
    }

  }

  Worker::Worker(ModuleDescription const& iMD, 
		 WorkerParams const& iWP) :
    stopwatch_(),
    timesRun_(),
    timesVisited_(),
    timesPassed_(),
    timesFailed_(),
    timesExcept_(),
    state_(Ready),
    md_(iMD),
    moduleCallingContext_(&md_),
    actions_(iWP.actions_),
    cached_exception_(),
    actReg_(),
    earlyDeleteHelper_(nullptr)
  {
  }

  Worker::~Worker() {
  }

  std::unique_ptr<OutputModuleCommunicator>
  Worker::createOutputModuleCommunicator() {
    return std::move(std::unique_ptr<OutputModuleCommunicator>{});
  }

  void Worker::setActivityRegistry(boost::shared_ptr<ActivityRegistry> areg) {
    actReg_ = areg;
  }

  void Worker::setEarlyDeleteHelper(EarlyDeleteHelper* iHelper) {
    earlyDeleteHelper_=iHelper;
  }
  
  void Worker::beginJob() {
    try {
      try {
        ModuleBeginJobSignalSentry cpp(actReg_.get(), md_);
        implBeginJob();
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginJob for module " << md_.moduleName() << "/'" << md_.moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }
  
  void Worker::endJob() {
    try {
      try {
        ModuleEndJobSignalSentry cpp(actReg_.get(), md_);
        implEndJob();
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endJob for module " << md_.moduleName() << "/'" << md_.moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::beginStream(StreamID id, StreamContext& streamContext) {
    try {
      try {
        streamContext.setTransition(StreamContext::Transition::kBeginStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ModuleBeginStreamSignalSentry beginSentry(actReg_.get(), streamContext, md_);
        implBeginStream(id);
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling beginStream for module " << md_.moduleName() << "/'" << md_.moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }
  
  void Worker::endStream(StreamID id, StreamContext& streamContext) {
    try {
      try {
        streamContext.setTransition(StreamContext::Transition::kEndStream);
        streamContext.setEventID(EventID(0, 0, 0));
        streamContext.setRunIndex(RunIndex::invalidRunIndex());
        streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
        streamContext.setTimestamp(Timestamp());
        ModuleEndStreamSignalSentry endSentry(actReg_.get(), streamContext, md_);
        implEndStream(id);
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      state_ = Exception;
      std::ostringstream ost;
      ost << "Calling endStream for module " << md_.moduleName() << "/'" << md_.moduleLabel() << "'";
      ex.addContext(ost.str());
      throw;
    }
  }

  void Worker::useStopwatch(){
    stopwatch_.reset(new RunStopwatch::StopwatchPointer::element_type);
  }
  
  void Worker::pathFinished(EventPrincipal& iEvent) {
    if(earlyDeleteHelper_) {
      earlyDeleteHelper_->pathFinished(iEvent);
    }
  }
  void Worker::postDoEvent(EventPrincipal& iEvent) {
    if(earlyDeleteHelper_) {
      earlyDeleteHelper_->moduleRan(iEvent);
    }
  }
}
