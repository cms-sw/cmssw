#ifndef FWCore_Framework_AnalyzerWorker_h
#define FWCore_Framework_AnalyzerWorker_h

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.18 2006/11/03 17:57:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class AnalyzerWorker : public Worker {
  public:
    AnalyzerWorker(std::auto_ptr<EDAnalyzer>,
		   ModuleDescription const&,
		   WorkerParams const&);
    virtual ~AnalyzerWorker();

  template <class ModType>
  static std::auto_ptr<EDAnalyzer> makeOne(ModuleDescription const& md,
					   WorkerParams const& wp);

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
    
    boost::shared_ptr<EDAnalyzer> analyzer_;
  };

  template <> 
  struct WorkerType<EDAnalyzer> {
    typedef EDAnalyzer ModuleType;
    typedef AnalyzerWorker worker_type;
  };

  template <class ModType>
  std::auto_ptr<EDAnalyzer> AnalyzerWorker::makeOne(ModuleDescription const&,
						    WorkerParams const& wp) {
    return std::auto_ptr<EDAnalyzer>(new ModType(*wp.pset_));
  }
}

#endif
