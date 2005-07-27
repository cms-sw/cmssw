
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.6 2005/07/23 07:32:57 wmtan Exp $
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
      md_(md),
      mod_(mod),
      actions_(wp.actions_) {
    assert(wp.reg_ != 0);
    mod_->setProductRegistry(*wp.reg_);
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
	  << md_;

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
	     << md_
	     << endl;
	throw;
    }
    catch(std::exception& e) {
	cerr << "An std::exception is going through OutputModule:\n"
	     << md_
	     << endl;
	throw;
    }
    catch(std::string& s) {
	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << md_ ;
    }
    catch(char const* c) {
	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << md_ ;
    }
    catch(...) {
	cerr << "An unknown Exception occured in\n" << md_;
	throw;
    }

    return rc;
  }

  void 
  OutputWorker::beginJob(EventSetup const&) {
  }

  void 
  OutputWorker::endJob() {
  }
   
}
