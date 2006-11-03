
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.12 2006/06/20 23:13:27 paterno Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm
{
  FilterWorker::FilterWorker(std::auto_ptr<EDFilter> ed,
			     const ModuleDescription& md,
			     const WorkerParams& wp):
   Worker(md, wp),
   filter_(ed)
  {
    filter_->registerProducts(filter_, wp.reg_, md, false);
  }

  FilterWorker::~FilterWorker()
  {
  }

  bool 
  FilterWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
			   CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Event e(ep,description());
    rc = filter_->doFilter(e, c, cpc);
    e.commit_();
    return rc;
  }

  void 
  FilterWorker::implBeginJob(EventSetup const& es) 
  {
    filter_->doBeginJob(es);
  }

  void 
  FilterWorker::implEndJob() 
  {
   filter_->doEndJob();
  }

  bool FilterWorker::implBeginRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    rc = filter_->doBeginRun(r,c,cpc);
    r.commit_();
    return rc;
  }

  bool FilterWorker::implEndRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    rc = filter_->doEndRun(r,c,cpc);
    r.commit_();
    return rc;
  }

  bool FilterWorker::implBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    rc = filter_->doBeginLuminosityBlock(lb,c,cpc);
    lb.commit_();
    return rc;
  }

  bool FilterWorker::implEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    rc = filter_->doEndLuminosityBlock(lb,c,cpc);
    lb.commit_();
    return rc;
  }

  std::string FilterWorker::workerType() const {
    return "EDFilter";
  }
  
}
