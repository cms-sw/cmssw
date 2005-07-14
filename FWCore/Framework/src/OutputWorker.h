#ifndef EDM_EDOUTPUTWORKED_INCLUDED
#define EDM_EDOUTPUTWORKED_INCLUDED

/*----------------------------------------------------------------------
  
OutputWorker: The OutputModule as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.

$Id: OutputWorker.h,v 1.4 2005/07/08 00:09:42 chrjones Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

#include "FWCore/Framework/src/Worker.h"

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

    virtual void beginJob( EventSetup const& ) ;
    virtual void endJob() ;
    
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
