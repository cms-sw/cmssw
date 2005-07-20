
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.4 2005/07/20 03:00:36 jbk Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <sstream>
#include <iostream>

using namespace std;

namespace edm
{

  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp):
    md_(md),
    producer_(ed),
    actions_(wp.actions_)
  {
  }

  ProducerWorker::~ProducerWorker()
  {
  }

  bool ProducerWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    bool rc = false;

    try
      {
	Event e(ep,md_);
	producer_->produce(e,c);
	e.commit_();
	rc=true;
      }
    catch(cms::Exception& e)
      {
	// should event id be included?
	e << "A cms::Exception is going through EDProducer:\n"
	  << md_;

	switch(actions_->find(e.rootCause()))
	  {
	  case actions::IgnoreCompletely:
	    {
	      rc=true;
	      cerr << "Producer ignored an exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  case actions::FailModule:
	    {
	      cerr << "Producer failed due to exception for event " << ep.id()
		   << "\nmessage from exception:\n" << e.what()
		   << endl;
	      break;
	    }
	  default: throw;
	  }
      }
    catch(seal::Error& e)
      {
	cerr << "A seal::Error is going through EDProducer:\n"
	     << md_
	     << endl;
	throw;
      }
    catch(std::exception& e)
      {
	cerr << "An std::exception is going through EDProducer:\n"
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
	cerr << "An unknown Exception occured in\n" << md_;
	throw;
      }

    return rc;
  }

  void ProducerWorker::beginJob( EventSetup const& es) 
  {
    producer_->beginJob(es);
  }

  void ProducerWorker::endJob() 
  {
    producer_->endJob();
  }

}
