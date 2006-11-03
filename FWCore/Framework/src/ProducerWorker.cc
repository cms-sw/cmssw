

/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.23 2006/10/31 23:54:01 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

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

  bool ProducerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
				  CurrentProcessingContext const* cpc) {

    bool rc = false;

    Event e(ep,description());
    producer_->doProduce(e, c, cpc);
    e.commit_();
    rc = true;
    return rc;
  }

  void ProducerWorker::implBeginJob(EventSetup const& es) {
    producer_->doBeginJob(es);
  }

  void ProducerWorker::implEndJob() {
    producer_->doEndJob();
  }
  
  bool ProducerWorker::implBeginRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    producer_->doBeginRun(r,c,cpc);
    r.commit_();
    rc = true;
    return rc;
  }

  bool ProducerWorker::implEndRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    producer_->doEndRun(r,c,cpc);
    r.commit_();
    rc = true;
    return rc;
  }

  bool ProducerWorker::implBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    producer_->doBeginLuminosityBlock(lb,c,cpc);
    lb.commit_();
    rc = true;
    return rc;
  }

  bool ProducerWorker::implEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    producer_->doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    rc = true;
    return rc;
  }

  std::string ProducerWorker::workerType() const {
    return "EDProducer";
  }
  
}
