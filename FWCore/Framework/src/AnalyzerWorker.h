#ifndef FWCore_Framework_AnalyzerWorker_h
#define FWCore_Framework_AnalyzerWorker_h

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.21 2008/01/11 20:30:08 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class AnalyzerWorker : public WorkerT<EDAnalyzer> {
  public:
    AnalyzerWorker(std::auto_ptr<EDAnalyzer>,
		   ModuleDescription const&,
		   WorkerParams const&);
    virtual ~AnalyzerWorker();

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
  struct WorkerType<EDAnalyzer> {
    typedef EDAnalyzer ModuleType;
    typedef AnalyzerWorker worker_type;
  };
}

#endif
