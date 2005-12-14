#ifndef Framework_Worker_h
#define Framework_Worker_h

/*----------------------------------------------------------------------
  
Worker: this is a basic scheduling unit - an abstract base class to
something that is really a producer or filter.

$Id: Worker.h,v 1.5 2005/09/08 10:57:35 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"

namespace edm {
  class ActionTable;

  class Worker
  {
  public:
    Worker(const ModuleDescription& iMD, const WorkerParams& iWP): md_(iMD),actions_(iWP.actions_) {}
    virtual ~Worker();
    bool doWork(EventPrincipal&, EventSetup const& c) ;
    void beginJob(EventSetup const&) ;
    void endJob();
    
    const ModuleDescription& description() const {return md_;}
   
  protected:
    virtual std::string workerType() const = 0;
    virtual bool implDoWork(EventPrincipal&, EventSetup const& c) = 0;
    virtual void implBeginJob(EventSetup const&) = 0;
    virtual void implEndJob() = 0;
    
  private:
    ModuleDescription md_;
    const ActionTable* actions_; // memory assumed to be managed elsewhere
  };

  template <class WT>
  struct WorkerType
  {
    // typedef int module_type;
    // typedef int worker_type;
  };

}
#endif
