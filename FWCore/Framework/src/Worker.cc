
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/Worker.h"

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
    actions_(iWP.actions_),
    cached_exception_(),
    actReg_()
  {
  }

  Worker::~Worker() {
  }

  void Worker::setActivityRegistry(boost::shared_ptr<ActivityRegistry> areg) {
    actReg_ = areg;
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
  
  void Worker::useStopwatch(){
    stopwatch_.reset(new RunStopwatch::StopwatchPointer::element_type);
  }
  
}
