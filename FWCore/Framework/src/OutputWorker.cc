
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.2 2005/04/05 16:24:12 paterno Exp $
----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/OutputModule.h"

#include "FWCore/CoreFramework/src/OutputWorker.h"

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
}
