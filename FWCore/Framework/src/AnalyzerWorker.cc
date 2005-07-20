
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.4 2005/07/20 03:00:36 jbk Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm
{
  AnalyzerWorker::AnalyzerWorker(std::auto_ptr<EDAnalyzer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp):
    md_(md),
    analyzer_(ed),
    actions_(wp.actions_)
  {
  }

  AnalyzerWorker::~AnalyzerWorker()
  {
  }

  bool AnalyzerWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    bool rc = false;
    try
      {
	Event e(ep,md_);
	analyzer_->analyze(e,c);
	rc = true;
      }
    catch(cms::Exception& e)
      {
	e << "A cms::Exception is going through EDAnalyzer:\n"
	  << md_;

	switch(actions_->find(e.rootCause()))
	  {
	  case actions::IgnoreCompletely:
	    {
	      rc=true;
	      cerr << "Analzer ignored an exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  case actions::FailModule:
	    {
	      cerr << "Analyzer failed due to exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  default: throw;
	  }
      }
    catch(seal::Error& e)
      {
	cerr << "A seal::Error is going through EDAnalyzer:\n"
	     << md_
	     << endl;
	throw;
      }
    catch(std::exception& e)
      {
	cerr << "An std::exception is going through EDAnalyzer:\n"
	     << md_
	     << endl;
	throw;
      }
    catch(std::string& s)
      {
	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << md_;
      }
    catch(const char* c)
      {
	throw cms::Exception("BadExceptionType","const char*") 
	  << "cstring = " << c << "\n"
	  << md_ ;
      }
    catch(...)
      {
	cerr << "An unknown Exception occured in\n" << md_;
	throw;
      }

    return rc;
  }

  void AnalyzerWorker::beginJob( EventSetup const& es) 
  {
    analyzer_->beginJob(es);
  }

  void AnalyzerWorker::endJob() 
  {
    analyzer_->endJob();
  }
}
