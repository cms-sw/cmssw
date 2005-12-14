
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.8 2005/09/08 10:57:35 chrjones Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
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
   Worker(md,wp),
   filter_(ed)
  {
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
    // a filter cannot write into the event, so commit is not needed
    // although we do know about what it asked for
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
