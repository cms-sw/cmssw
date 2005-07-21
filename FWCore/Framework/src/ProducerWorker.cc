
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.5 2005/07/20 04:11:41 jbk Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/ProductDescription.h"
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

    md_(md),
    producer_(ed),
    actions_(wp.actions_)
  {
    
    const EDProducer::TypeLabelList& plist= producer_->getTypeLabelList();

    EDProducer::TypeLabelList::const_iterator p;
    // loop on products declared by the User Module
    // and register them with ProductRegistry
    for(p=plist.begin(); p!=plist.end(); ++p){
           
 
      ProductDescription pdesc;
      pdesc.product_id = EDP_ID();
      pdesc.module = md;
      pdesc.full_product_type_name =  p->first.userClassName();
      pdesc.friendly_product_type_name =  p->first.friendlyClassName();
      pdesc.product_instance_name =   p->second;
      /*
      ProductDescription pdesc(EDP_ID(),md,
			       p->first.userClassName(),
			       p->first.friendlyClassName(), 
			       p->second);
      */
      wp.reg_->addProduct(pdesc);
    }//for

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
