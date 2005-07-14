
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm
{
  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 const ModuleDescription& md):
    md_(md),
    producer_(ed)
  {
  }

  ProducerWorker::~ProducerWorker()
  {
  }

  bool ProducerWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    Event e(ep,md_);
    producer_->produce(e,c);
    e.commit_();
    return true;
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
