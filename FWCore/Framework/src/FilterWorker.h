#ifndef FWCore_Framework_FilterWorker_h
#define FWCore_Framework_FilterWorker_h

/*----------------------------------------------------------------------
  
FilterWorker: The EDFilter as the schedule sees it.  The job of
this object is to call the filter.
According to our current definition, a single filter can only
appear in one worker.

$Id: FilterWorker.h,v 1.18 2007/06/05 04:02:32 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class FilterWorker : public Worker {
  public:
    FilterWorker(std::auto_ptr<EDFilter>,
		 ModuleDescription const&,
		 WorkerParams const&);
    virtual ~FilterWorker();

    template <class ModType>
    static std::auto_ptr<EDFilter> makeOne(ModuleDescription const&,
					   WorkerParams const&);

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
    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual std::string workerType() const;

    boost::shared_ptr<EDFilter> filter_;
  };

  template <> 
  struct WorkerType<EDFilter> {
    typedef EDFilter ModuleType;
    typedef FilterWorker worker_type;
  };

  template <class ModType>
  std::auto_ptr<EDFilter> FilterWorker::makeOne(ModuleDescription const& md,
						WorkerParams const& wp) {
    std::auto_ptr<ModType> filter = std::auto_ptr<ModType>(new ModType(*wp.pset_));
    filter->setModuleDescription(md);
    return std::auto_ptr<EDFilter>(filter.release());
  }

}

#endif
