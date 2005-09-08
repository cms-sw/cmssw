#ifndef Framework_Worker_h
#define Framework_Worker_h

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.4 2005/09/01 04:30:28 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm {

  class Worker
  {
  public:
    Worker(const ModuleDescription& iMD): md_(iMD) {}
    virtual ~Worker();
    virtual bool doWork(EventPrincipal&, EventSetup const& c) = 0;
    virtual void beginJob(EventSetup const&) = 0;
    virtual void endJob() = 0;
    
    const ModuleDescription& description() const {return md_;}
  private:
    ModuleDescription md_;
  };

  template <class WT>
  struct WorkerType
  {
    // typedef int module_type;
    // typedef int worker_type;
  };

}
#endif
