
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
    
  }

  Worker::Worker(ModuleDescription const& iMD, 
		 WorkerParams const& iWP) :
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
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
        ModuleBeginJobSignalSentry cpp(actReg_.get(), md_);
	implBeginJob();
    }
    catch(cms::Exception& e) {
	LogError("BeginJob")
	  << "A cms::Exception is going through " << workerType() << ":\n";
	state_ = Exception;
	e << "A cms::Exception is going through " << workerType() << ":\n";
	exceptionContext(md_, e);
	throw e;
    }
    catch(std::bad_alloc& bda) {
	LogError("BeginJob")
	  << "A std::bad_alloc is going through " << workerType() << ":\n"
	  << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadAlloc));
	*cached_exception_
	  << "A std::bad_alloc exception occurred during a call to the module ";
	exceptionContext(md_, *cached_exception_)
	  << "The job has probably exhausted the virtual memory available to the process.\n";
	throw *cached_exception_;
    }
    catch(std::exception& e) {
	LogError("BeginJob")
	  << "A std::exception is going through " << workerType() << ":\n"
	  << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::StdException));
	*cached_exception_
	  << "A std::exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n" << e.what();
	throw *cached_exception_;
    }
    catch(std::string& s) {
	LogError("BeginJob") 
	  << "module caught a std::string during endJob\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "std::string"));
	*cached_exception_
	  << "A std::string thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n string = " << s;
	throw *cached_exception_;
    }
    catch(char const* c) {
	LogError("BeginJob") 
	  << "module caught a const char* during endJob\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "const char *"));
	*cached_exception_
	  << "A const char* thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n const char* = " << c << "\n";
	throw *cached_exception_;
    }
    catch(...) {
	LogError("BeginJob")
	  << "An unknown Exception occurred in\n" << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::Unknown, "repeated"));
	*cached_exception_
	  << "An unknown occurred during a previous call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n";
	throw *cached_exception_;
    }
  }
  
  void Worker::endJob() {
    try {
        ModuleEndJobSignalSentry cpp(actReg_.get(), md_);
	implEndJob();
    }
    catch(cms::Exception& e) {
	LogError("EndJob")
	  << "A cms::Exception is going through " << workerType() << ":\n";
	state_ = Exception;
	e << "A cms::Exception is going through " << workerType() << ":\n";
	exceptionContext(md_, e);
	throw e;
    }
    catch(std::bad_alloc& bda) {
	LogError("EndJob")
	  << "A std::bad_alloc is going through " << workerType() << ":\n"
	  << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadAlloc));
	*cached_exception_
	  << "A std::bad_alloc exception occurred during a call to the module ";
	exceptionContext(md_, *cached_exception_)
	  << "The job has probably exhausted the virtual memory available to the process.\n";
	throw *cached_exception_;
    }
    catch(std::exception& e) {
	LogError("EndJob")
	  << "A std::exception is going through " << workerType() << ":\n"
	  << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::StdException));
	*cached_exception_
	  << "A std::exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n" << e.what();
	throw *cached_exception_;
    }
    catch(std::string& s) {
	LogError("EndJob") 
	  << "module caught a std::string during endJob\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "std::string"));
	*cached_exception_
	  << "A std::string thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n string = " << s;
	throw *cached_exception_;
    }
    catch(char const* c) {
	LogError("EndJob") 
	  << "module caught a const char* during endJob\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::BadExceptionType, "const char *"));
	*cached_exception_
	  << "A const char* thrown as an exception occurred during a call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n"
	  << "Previous information:\n const char* = " << c << "\n";
	throw *cached_exception_;
    }
    catch(...) {
	LogError("EndJob")
	  << "An unknown Exception occurred in\n" << description() << "\n";
	state_ = Exception;
	cached_exception_.reset(new edm::Exception(errors::Unknown, "repeated"));
	*cached_exception_
	  << "An unknown occurred during a previous call to the module ";
        exceptionContext(md_, *cached_exception_) << "and cannot be repropagated.\n";
	throw *cached_exception_;
    }

  }
  
}
