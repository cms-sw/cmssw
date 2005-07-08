
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/src/ProducerWorker.h"

#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/EDProducer.h"

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
