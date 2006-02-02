
/*----------------------------------------------------------------------
$Id: Worker.cc,v 1.5 2006/02/02 20:05:36 llista Exp $
----------------------------------------------------------------------*/

#include <iostream>
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Log.h"

#include "boost/signal.hpp"

namespace edm
{
  namespace
  {
    class CallPrePost
    {
    public:
      CallPrePost(Worker::Sigs& s, ModuleDescription& md):s_(&s),md_(&md)
      { s_->preModuleSignal(*md_); }
      ~CallPrePost()
      { s_->postModuleSignal(*md_); }
    private:
      Worker::Sigs* s_;
      ModuleDescription* md_;
    };
  }
  
  Worker::Worker(const ModuleDescription& iMD, 
		 const WorkerParams& iWP):
    timesVisited_(),
    timesRun_(),
    timesFailed_(),
    timesPass_(),
    timesExcept_(),
    state_(Ready),
    md_(iMD),
    actions_(iWP.actions_),
    cached_exception_()
  {
  }

  Worker::~Worker()
  {
  }

  void Worker::connect(ActivityRegistry::PreModule& pre,
		       ActivityRegistry::PostModule& post)
  {
    sigs_.preModuleSignal.connect(pre);
    sigs_.postModuleSignal.connect(post);
  }

  bool Worker::doWork(EventPrincipal& ep, EventSetup const& es)
  {
    using namespace std;

    bool rc = false;
    ++timesVisited_;

    switch(state_)
      {
      case Ready: break;
      case Pass: return true;
      case Fail: return false;
      case Exception:
	{
	  // rethrow the cached exception again
	  // only cms::Exceptions can be cached and contributing to
	  // actions or processing routing.  It seems impossible to
	  // get here a second time until a cms::Exception has been 
	  // thrown prviously.
	  LogWarning("repeat") << "A module has been invoked a second "
			       << "time even though it caught an "
			       << "exception during the previous "
			       << "invocation.\n"
			       << "This may be an indication of a "
			       << "configuration problem.\n";

	  throw *cached_exception_;
	}
      }

    try
      {
	CallPrePost cpp(sigs_,md_);
	++timesRun_;

	rc = implDoWork(ep,es);

	if(rc)
	  {
	    ++timesPass_;
	    state_ = Pass;
	  }
	else
	  {
	    ++timesFailed_;
	    state_ = Fail;
	  }
      }

    catch(cms::Exception& e)
      {
      
	// NOTE: the warning printed as a result of ignoring or failing
	// a module will only be printed during the full true processing
	// pass of this module

	switch(actions_->find(e.rootCause()))
	  {
	  case actions::IgnoreCompletely:
	    {
	      rc=true;
	      state_=Pass;
	      LogWarning("IgnoreCompletely")
		<< "Module ignored an exception\n";
	      break;
	    }

	  case actions::FailModule:
	    {
	      LogWarning("FailModule")
		<< "Module failed an event due to exception\n";
	      state_=Fail;
	      break;
	    }
	    
	  default:
	    {
	      // should event id be included?
	      LogError("ModuleFailure")
		<< "Module received cms::Exception.\n";

	      // we should not need to include the event/run/module names
	      // the exception because the error logger will pick this
	      // up automatically.  I'm leaving it in until this is 
	      // verified

	      // here we simply add a small amount of data to the
	      // exception to add some context, we could have rethrown
	      // is as something else and embedded with this exception
	      // as an argument to the constructor.

	      ++timesExcept_;
	      e << "cms::Exception going through module\n"
		<< md_.moduleName_ << "/" << md_.moduleLabel_ 
		<< " " << ep.id() << "\n";
	      state_ = Exception;
	      cached_exception_.reset(new cms::Exception(e));
	      throw;
	    }
	  }
	LogError(e.category()) << e.what() << "\n";
      }
    
    catch(seal::Error& e)
      {
	LogError("ModuleFailure")
	  << "Module received seal::Error.\n";
	++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new cms::Exception("repeated"));
	*cached_exception_
	  << "A seal::Error occurred during a previous call to this "
	  << "module and cannot be repropagated.\n"
	  << "Previous information:\n" << e.explainSelf();
	  
	throw *cached_exception_;
      }
    catch(std::exception& e)
      {
	LogError("ModuleFailure")
	  << "Module received std::exception.\n";
	++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new cms::Exception("repeated"));
	*cached_exception_
	  << "An std::exception occurred during a previous call to this "
	  << "module and cannot be repropagated.\n"
	  << "Previous information:\n" << e.what();
	throw *cached_exception_;
      }
    catch(std::string& s)
      {
	LogError("BadExceptionType")
	  << "Module received std::string as an exception.\n"
	  << "string = " << s << "\n";

	++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new cms::Exception("BadExceptionType","std::string"));
	*cached_exception_
	  << "string = " << s << "\n"
	  << md_.moduleName_ << "/" << md_.moduleLabel_ 
	  << ep.id() << "\n";
	throw *cached_exception_;
      }
    catch(const char* c)
      {
	LogError("BadExceptionType")
	  << "Module received const char* as an exception.\n"
	  << "const char* = " << c << "\n";

	++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new cms::Exception("BadExceptionType","const char*"));

	*cached_exception_
	  << "cstring = " << c << "\n"
	  << md_.moduleName_ << "/" << md_.moduleLabel_ 
	  << ep.id() << "\n";
	throw *cached_exception_;
      }
    catch(...)
      {
	LogError("BadExceptionType")
	  << "Module received an unknown exception.\n";

	++timesExcept_;
	state_ = Exception;
	cached_exception_.reset(new cms::Exception("repeated"));

	*cached_exception_
	  << "An unknown exception occurred during a previous call to this "
	  << "module and cannot be repropagated.\n";
	throw;
      }

    return rc;
  }

  void Worker::beginJob(EventSetup const& es)
  {
    using namespace std;
    
    try
      {
	implBeginJob(es);
      }
    catch(cms::Exception& e)
      {
	// should event id be included?
	LogError("BeginJob")
	  << "A cms::Exception is going through "<< workerType()<<":\n";

	e << "A cms::Exception is going through "<< workerType()<<":\n"
	  << description();
	throw;
      }
    catch(seal::Error& e)
      {
	LogError("BeginJob")
	  << "A seal::Error is going through "<< workerType()<<":\n"
	  << description() << "\n";
	throw;
      }
    catch(std::exception& e)
      {
	LogError("BeginJob")
	  << "An std::exception is going through "<< workerType()<<":\n"
	  << description() << "\n";
	throw;
      }
    catch(std::string& s)
      {
	LogError("BeginJob") 
	  << "module caught an std::string during beginJob\n";

	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << description() << "\n" ;
      }
    catch(const char* c)
      {
	LogError("BeginJob") 
	  << "module caught an const char* during beginJob\n";

	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << description() ;
      }
    catch(...)
      {
	LogError("BeginJob")
	  << "An unknown Exception occured in\n" << description() << "\n";
	throw;
      }
  }
  
  void Worker::endJob()
  {
    using namespace std;
    
    try
      {
	implEndJob();
      }
    catch(cms::Exception& e)
      {
	LogError("EndJob")
	  << "A cms::Exception is going through "<< workerType()<<":\n";

	// should event id be included?
	e << "A cms::Exception is going through "<< workerType()<<":\n"
	  << description();
	throw;
      }
    catch(seal::Error& e)
      {
	LogError("EndJob")
	  << "A seal::Error is going through "<< workerType()<<":\n"
	  << description() << "\n";
	throw;
      }
    catch(std::exception& e)
      {
	LogError("EndJob")
	  << "An std::exception is going through "<< workerType()<<":\n"
	  << description() << "\n";
	throw;
      }
    catch(std::string& s)
      {
	LogError("EndJob") 
	  << "module caught an std::string during endJob\n";

	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << description() << "\n";
      }
    catch(const char* c)
      {
	LogError("EndJob") 
	  << "module caught an const char* during endJob\n";

	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << description() << "\n";
      }
    catch(...)
      {
	LogError("EndJob")
	  << "An unknown Exception occured in\n" << description() << "\n";
	throw;
      }
  }
  
}
