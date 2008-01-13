

/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.27 2008/01/11 20:30:08 wmtan Exp $
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
    WorkerT<EDProducer>(ed, md, wp)
  {
    module().registerProducts(moduleSharedPtr(), wp.reg_, md, true);
  }

  ProducerWorker::~ProducerWorker() {
  }

  bool
  ProducerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
			     BranchActionType,
			     CurrentProcessingContext const* cpc) {

    bool rc = false;

    Event e(ep,description());
    module().doProduce(e, c, cpc);
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
    if (bat == BranchActionBegin) module().doBeginRun(r,c,cpc);
    else module().doEndRun(r,c,cpc);
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
    if (bat == BranchActionBegin) module().doBeginLuminosityBlock(lb,c,cpc);
    else module().doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    rc = true;
    return rc;
  }

  std::string ProducerWorker::workerType() const {
    return "EDProducer";
  }
  
}
