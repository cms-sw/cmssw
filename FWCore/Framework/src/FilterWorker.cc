
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.14 2007/03/04 06:10:25 wmtan Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

namespace edm {
  FilterWorker::FilterWorker(std::auto_ptr<EDFilter> ed,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
   Worker(md, wp),
   filter_(ed)
  {
    filter_->registerProducts(filter_, wp.reg_, md, false);
  }

  FilterWorker::~FilterWorker() {
  }

  bool 
  FilterWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    Event e(ep,description());
    rc = filter_->doFilter(e, c, cpc);
    e.commit_();
    return rc;
  }

  bool
  FilterWorker::implDoWork(RunPrincipal& rp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    Run r(rp,description());
    if (bat == BranchActionBegin) rc = filter_->doBeginRun(r,c,cpc);
    else rc = filter_->doEndRun(r,c,cpc);
    r.commit_();
    return rc;
  }

  bool
  FilterWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    if (bat == BranchActionBegin) rc = filter_->doBeginLuminosityBlock(lb,c,cpc);
    else rc = filter_->doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    return rc;
  }

  void 
  FilterWorker::implBeginJob(EventSetup const& es) {
    filter_->doBeginJob(es);
  }

  void 
  FilterWorker::implEndJob() {
   filter_->doEndJob();
  }

  std::string FilterWorker::workerType() const {
    return "EDFilter";
  }
  
}
