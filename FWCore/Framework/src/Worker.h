#ifndef EDM_WORKER_HH
#define EDM_WORKER_HH

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.2 2005/07/08 00:09:42 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class Worker
  {
  public:
    virtual ~Worker();
    virtual bool doWork(EventPrincipal&, EventSetup const& c) = 0;
    virtual void beginJob( EventSetup const& ) = 0;
    virtual void endJob() = 0;
  };

  template <class WT>
  struct WorkerType
  {
    // typedef int module_type;
    // typedef int worker_type;
  };

}
#endif
