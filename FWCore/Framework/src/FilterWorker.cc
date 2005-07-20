
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.5 2005/07/20 03:00:36 jbk Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm
{
  FilterWorker::FilterWorker(std::auto_ptr<EDFilter> ed,
			     const ModuleDescription& md,
			     const WorkerParams& wp):
   md_(md),
   filter_(ed),
   actions_(wp.actions_)
  {
  }

  FilterWorker::~FilterWorker()
  {
  }

  bool 
  FilterWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    bool rc = false;
    try
      {
	Event e(ep,md_);
	rc = filter_->filter(e, c);
	// a filter cannot write into the event, so commit is not needed
	// although we do know about what it asked for
      }
    catch(cms::Exception& e)
      {
	e << "A cms::Exception is going through EDFilter:\n"
	  << md_;

	switch(actions_->find(e.rootCause()))
	  {
	  case actions::IgnoreCompletely:
	    {
	      rc=true;
	      cerr << "Filter ignored an exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  case actions::FailModule:
	    {
	      cerr << "Filter failed due to exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  default: throw;
	  }

      }
    catch(seal::Error& e)
      {
	cerr << "A seal::Error is going through EDFilter:\n"
	     << md_
	     << endl;
	throw;
      }
    catch(std::exception& e)
      {
	cerr << "An std::exception is going through EDFilter:\n"
	     << md_
	     << endl;
	throw;
      }
    catch(std::string& s)
      {
	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << md_ ;
      }
    catch(const char* c)
      {
	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << md_ ;
      }
    catch(...)
      {
	cerr << "An unknown Exception occured in:\n" << md_;
	throw;
      }

    return rc;
  }

  void 
  FilterWorker::beginJob( EventSetup const& es) 
  {
    filter_->beginJob(es);
  }

  void 
  FilterWorker::endJob() 
  {
   filter_->endJob();
  }

}
