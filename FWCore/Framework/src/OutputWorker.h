#ifndef EDM_EDOUTPUTWORKED_INCLUDED
#define EDM_EDOUTPUTWORKED_INCLUDED

/*----------------------------------------------------------------------
  
OutputWorker: The OutputModule as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.

$Id: OutputWorker.h,v 1.2 2005/06/23 05:23:10 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/CoreFramework/interface/OutputModule.h"
#include "FWCore/CoreFramework/interface/ModuleDescription.h"

#include "FWCore/CoreFramework/src/Worker.h"

namespace edm
{

  class OutputWorker : public Worker
  {
  public:
    OutputWorker(std::auto_ptr<OutputModule> mod, 
		 const ModuleDescription&);

    virtual ~OutputWorker();

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    ModuleDescription               md_;
    boost::shared_ptr<OutputModule> mod_;
  };

  template <> 
  struct WorkerType<OutputModule>
  {
    typedef OutputModule ModuleType;
    typedef OutputWorker worker_type;
  };
}

#endif
