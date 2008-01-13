
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.18 2008/01/11 20:30:08 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/src/WorkerParams.h"

namespace edm {
  FilterWorker::FilterWorker(std::auto_ptr<EDFilter> ed,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
   WorkerT<EDFilter>(ed, md, wp)
  {
    module().registerProducts(moduleSharedPtr(), wp.reg_, md, false);
  }

  FilterWorker::~FilterWorker() {
  }

  bool 
  FilterWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    Event e(ep,description());
    rc = module().doFilter(e, c, cpc);
    e.commit_();
    return rc;
  }

  bool
  FilterWorker::implDoWork(RunPrincipal& rp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    Run r(rp,description());
    if (bat == BranchActionBegin) rc = module().doBeginRun(r,c,cpc);
    else rc = module().doEndRun(r,c,cpc);
    r.commit_();
    return rc;
  }

  bool
  FilterWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    if (bat == BranchActionBegin) rc = module().doBeginLuminosityBlock(lb,c,cpc);
    else rc = module().doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    return rc;
  }

  std::string FilterWorker::workerType() const {
    return "EDFilter";
  }
  
}
