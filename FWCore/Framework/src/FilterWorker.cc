
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.11 2006/04/20 22:33:22 wmtan Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
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

  std::string FilterWorker::workerType() const {
    return "EDFilter";
  }
  
}
