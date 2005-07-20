#ifndef EDM_EDFILTERWORKER_INCLUDED
#define EDM_EDFILTERWORKER_INCLUDED

/*----------------------------------------------------------------------
  
FilterWorker: The EDFilter as the schedule sees it.  The job of
this object is to call the filter.
According to our current definition, a single filter can only
appear in one worker.

$Id: FilterWorker.h,v 1.5 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm
{
  class EventPrincipal;
  class ModuleDescription;
  class EDFilter;
  class ActionTable;
  class WorkerParams;

  class FilterWorker : public Worker
  {
  public:
    FilterWorker(std::auto_ptr<EDFilter>,
		 const ModuleDescription&,
		 const WorkerParams&);
    virtual ~FilterWorker();

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);
    virtual void beginJob( EventSetup const& ) ;
    virtual void endJob() ;

    ModuleDescription md_;
    boost::shared_ptr<EDFilter> filter_;
    const ActionTable* actions_; // memory assumed to be managed elsewhere
  };

  template <> 
  struct WorkerType<EDFilter>
  {
    typedef EDFilter ModuleType;
    typedef FilterWorker worker_type;
  };

}

#endif
