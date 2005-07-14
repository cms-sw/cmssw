
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/OutputModule.h"

#include "FWCore/Framework/src/OutputWorker.h"

namespace edm
{
  OutputWorker::OutputWorker(std::auto_ptr<OutputModule> mod,
			     const ModuleDescription& md):
    md_(md),
    mod_(mod)
  {
  }

  OutputWorker::~OutputWorker()
  {
  }

  bool 
  OutputWorker::doWork(EventPrincipal& ep, EventSetup const&)
  {
    // EventSetup is not (yet) used. Should it be passed to the
    // OutputModule?
    mod_->write(ep);
    return true;
  }

  void 
  OutputWorker::beginJob( EventSetup const& ) 
  {
  }

  void 
  OutputWorker::endJob() 
  {
  }
   
}
