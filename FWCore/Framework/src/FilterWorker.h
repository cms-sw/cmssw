#ifndef FWCore_Framework_FilterWorker_h
#define FWCore_Framework_FilterWorker_h

/*----------------------------------------------------------------------
  
FilterWorker: The EDFilter as the schedule sees it.  The job of
this object is to call the filter.
According to our current definition, a single filter can only
appear in one worker.

$Id: FilterWorker.h,v 1.20 2008/01/11 20:30:08 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class FilterWorker : public WorkerT<EDFilter> {
  public:
    FilterWorker(std::auto_ptr<EDFilter>,
		 ModuleDescription const&,
		 WorkerParams const&);
    virtual ~FilterWorker();

  private:
    virtual bool implDoWork(EventPrincipal& e, EventSetup const& c,
			    BranchActionType,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(RunPrincipal& rp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);

    virtual std::string workerType() const;
  };

  template <> 
  struct WorkerType<EDFilter> {
    typedef EDFilter ModuleType;
    typedef FilterWorker worker_type;
  };
}

#endif
