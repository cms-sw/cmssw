
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.13 2005/09/08 10:57:35 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <sstream>

using namespace std;

namespace edm
{

  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp):
    Worker(md),
    producer_(ed),
    actions_(wp.actions_) {
    
    EDProducer::TypeLabelList const& plist = producer_->typeLabelList();
    if (plist.empty()) {
      throw edm::Exception(errors::NoProductSpecified,"Producer")
        << "Module " << md.moduleName_
        << " did not specify that it produces a product.\n"
        << "The module constructor must call 'produces<T>(instanceName)'"
        << " for each product it produces.\nT is the product type.\n"
	<< "'instanceName' is an optional string used to distinguish" 
        << " multiple products of the same type.";
    }

    EDProducer::TypeLabelList::const_iterator p;
    // loop on products declared by the User Module
    // and register them with ProductRegistry
    for(p=plist.begin(); p!=plist.end(); ++p) {
           
 
     BranchDescription pdesc(md,
       p->typeID_.userClassName(),
       p->typeID_.friendlyClassName(), 
       p->productInstanceName_,
       p->productPtr_);
       wp.reg_->addProduct(pdesc);
     }//for

  }

  ProducerWorker::~ProducerWorker() {
  }

  bool ProducerWorker::doWork(EventPrincipal& ep, EventSetup const& c) {
    bool rc = false;

    try {
      Event e(ep,description());
      producer_->produce(e,c);
      e.commit_();
      rc=true;
    }
    catch(cms::Exception& e) {
	// should event id be included?
	e << "A cms::Exception is going through EDProducer:\n"
	  << description();

	switch(actions_->find(e.rootCause())) {
	  case actions::IgnoreCompletely: {
	    rc=true;
	    cerr << "Producer ignored an exception for event " << ep.id()
	         << "\nmessage from exception:\n" << e.what()
	         << endl;
	    break;
	  }
	  case actions::FailModule: {
	    cerr << "Producer failed due to exception for event " << ep.id()
		 << "\nmessage from exception:\n" << e.what()
		 << endl;
	    break;
	  }
	  default: throw;
	}
    }
    catch(seal::Error& e) {
	cerr << "A seal::Error is going through EDProducer:\n"
	     << description()
	     << endl;
	throw;
    }
    catch(std::exception& e) {
	cerr << "An std::exception is going through EDProducer:\n"
	     << description()
	     << endl;
	throw;
    }
    catch(std::string& s) {
	throw cms::Exception("BadExceptionType","std::string") 
	  << "string = " << s << "\n"
	  << description() ;
    }
    catch(const char* c) {
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

  void ProducerWorker::beginJob(EventSetup const& es) {
    producer_->beginJob(es);
  }

  void ProducerWorker::endJob() {
    producer_->endJob();
  }
}
