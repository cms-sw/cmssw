#ifndef EDM_EDFILTERWORKER_INCLUDED
#define EDM_EDFILTERWORKER_INCLUDED

/*----------------------------------------------------------------------
  
FilterWorker: The EDFilter as the schedule sees it.  The job of
this object is to call the filter.
According to our current definition, a single filter can only
appear in one worker.

$Id: FilterWorker.h,v 1.6 2005/07/20 03:00:36 jbk Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"

namespace edm
{
  class EventPrincipal;
  class ModuleDescription;
  class EDFilter;
  class ActionTable;
  class ParameterSet;

  class FilterWorker : public Worker
  {
  public:
    FilterWorker(std::auto_ptr<EDFilter>,
		 const ModuleDescription&,
		 const WorkerParams&);
    virtual ~FilterWorker();

    template <class ModType>
    static std::auto_ptr<EDFilter> makeOne(const ModuleDescription&,
					   const WorkerParams&);

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

  template <class ModType>
  std::auto_ptr<EDFilter> FilterWorker::makeOne(const ModuleDescription& md,
						const WorkerParams& wp)
  {
    return std::auto_ptr<EDFilter>(new ModType(*wp.pset_));
  }

}

#endif
