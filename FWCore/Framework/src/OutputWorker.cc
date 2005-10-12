
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.10 2005/09/08 10:57:35 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/src/OutputWorker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm {
  OutputWorker::OutputWorker(std::auto_ptr<OutputModule> mod,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
      Worker(md),
      mod_(mod),
      actions_(wp.actions_) {
  }

  OutputWorker::~OutputWorker() {
  }

  bool 
  OutputWorker::doWork(EventPrincipal& ep, EventSetup const&) {
    // EventSetup is not (yet) used. Should it be passed to the
    // OutputModule?
    bool rc = false;

    try {
	mod_->write(ep);
	rc=true;
    }
    catch(cms::Exception& e) {
	e << "A cms::Exception is going through OutputModule:\n"
	  << description();

	switch(actions_->find(e.rootCause())) {
	  case actions::IgnoreCompletely: {
	      rc=true;
	      cerr << "Output module ignored exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	  }
	  case actions::FailModule: {
	      cerr << "Output module failed due to exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	  }
	  default: throw;
	}
    }
    catch(seal::Error& e) {
	cerr << "A seal::Error is going through OutputModule:\n"
	     << description()
	     << endl;
	throw;
    }
    catch(std::exception& e) {
	cerr << "An std::exception is going through OutputModule:\n"
	     << description()
	     << endl;
	throw;
    }
    catch(std::string& s) {
	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << description() ;
    }
    catch(char const* c) {
	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << description() ;
    }
    catch(...) {
	cerr << "An unknown Exception occured in\n" << description();
	throw;
    }

    return rc;
  }

  void 
  OutputWorker::beginJob(EventSetup const& es) {
    mod_->beginJob(es);
  }

  void 
  OutputWorker::endJob() {
    mod_->endJob();
  }
   
}
