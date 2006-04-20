

/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.20 2006/04/19 01:48:06 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm
{
  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp) :
    Worker(md,wp),
    producer_(ed) {
    producer_->registerProducts(producer_, wp.reg_, md, true);
  }

  ProducerWorker::~ProducerWorker() {
  }

  bool ProducerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c) {
    bool rc = false;

    Event e(ep,description());
    producer_->produce(e, c);
    e.commit_();
    rc = true;
    return rc;
  }

  void ProducerWorker::implBeginJob(EventSetup const& es) {
    producer_->beginJob(es);
  }

  void ProducerWorker::implEndJob() {
    producer_->endJob();
  }
  
  std::string ProducerWorker::workerType() const {
    return "EDProducer";
  }
  
}
