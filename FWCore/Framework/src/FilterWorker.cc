
/*----------------------------------------------------------------------
$Id: FilterWorker.cc,v 1.3 2005/07/08 00:09:42 chrjones Exp $
----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Framework/src/FilterWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ModuleDescription.h"


namespace edm
{
  FilterWorker::FilterWorker(std::auto_ptr<EDFilter> ed,
			     const ModuleDescription& md):
   md_(md),
   filter_(ed)
  {
  }

  FilterWorker::~FilterWorker()
  {
  }

  bool 
  FilterWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    Event e(ep,md_);
    return filter_->filter(e, c);
    // a filter cannot write into the event, so commit is not needed
    // although we do know about what it asked for
  }

  void 
  FilterWorker::beginJob( EventSetup const& es) 
  {
    filter_->beginJob(es);
  }

  void 
  FilterWorker::endJob() 
  {
   filter_->endJob();
  }

}
