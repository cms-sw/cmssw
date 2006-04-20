
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.10 2006/02/08 00:44:25 wmtan Exp $
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
  FilterWorker::implDoWork(EventPrincipal& ep, EventSetup const& c)
  {
    bool rc = false;
    Event e(ep,description());
    rc = filter_->filter(e, c);
    e.commit_();
    return rc;
  }

  void 
  FilterWorker::implBeginJob(EventSetup const& es) 
  {
    filter_->beginJob(es);
  }

  void 
  FilterWorker::implEndJob() 
  {
   filter_->endJob();
  }

  std::string FilterWorker::workerType() const {
    return "EDFilter";
  }
  
}
