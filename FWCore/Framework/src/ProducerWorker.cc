

/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.25 2007/06/05 04:02:32 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/WorkerParams.h"

namespace edm {
  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 ModuleDescription const& md,
				 WorkerParams const& wp) :
    Worker(md,wp),
    producer_(ed)
  {
    producer_->registerProducts(producer_, wp.reg_, md, true);
  }

  ProducerWorker::~ProducerWorker() {
  }

  bool
  ProducerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
			     BranchActionType,
			     CurrentProcessingContext const* cpc) {

    bool rc = false;

    Event e(ep,description());
    producer_->doProduce(e, c, cpc);
    e.commit_();
    rc = true;
    return rc;
  }

  bool
  ProducerWorker::implDoWork(RunPrincipal& rp, EventSetup const& c,
			     BranchActionType bat,
			     CurrentProcessingContext const* cpc) {
    bool rc = false;
    Run r(rp,description());
    if (bat == BranchActionBegin) producer_->doBeginRun(r,c,cpc);
    else producer_->doEndRun(r,c,cpc);
    r.commit_();
    rc = true;
    return rc;
  }

  bool
  ProducerWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			     BranchActionType bat,
			     CurrentProcessingContext const* cpc) {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    if (bat == BranchActionBegin) producer_->doBeginLuminosityBlock(lb,c,cpc);
    else producer_->doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    rc = true;
    return rc;
  }

  void ProducerWorker::implBeginJob(EventSetup const& es) {
    producer_->doBeginJob(es);
  }

  void ProducerWorker::implEndJob() {
    producer_->doEndJob();
  }
  
  std::string ProducerWorker::workerType() const {
    return "EDProducer";
  }
  
}
